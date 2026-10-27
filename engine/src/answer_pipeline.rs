use crate::api_types::{
    AnswerClaimPlanReceipt, AnswerEvidenceAtomReceipt, AnswerEvidencePlanReceipt,
    AnswerEvidenceRejectionReceipt, AnswerEvidenceSourceAnchor, AnswerOperatorPlanReceipt,
    AnswerPipelineReceipt, AnswerPipelineStageReceipt, AnswerQueryPlanReceipt,
    AnswerSlotCoverageReceipt, CoverageReport, RetrievedItem,
};
use crate::memory_models::MemoryRecord;
use chrono::Datelike;
use regex::Regex;
use serde::Serialize;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

const PIPELINE_VERSION: &str = "question-evidence-claim-answer-v26-relevance-tiers";
pub(crate) const PERSISTED_EVIDENCE_PROJECTOR_VERSION: &str =
    "typed-evidence-projector-v13-addressee-reactions";
const MAX_RECEIPT_ATOMS: usize = 24;
const MAX_REJECTION_RECEIPTS: usize = 64;
const MAX_RUNTIME_ATOMS: usize = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EvidenceKind {
    Statement,
    Event,
    SessionDate,
    Entity,
    NativeMedia,
    SystemMetadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GroundingRole {
    Original,
    Context,
    ExpansionOnly,
}

impl GroundingRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Original => "original",
            Self::Context => "context",
            Self::ExpansionOnly => "expansion_only",
        }
    }
}

impl EvidenceKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Statement => "body_fact",
            Self::Event => "event",
            Self::SessionDate => "session_date",
            Self::Entity => "entity",
            Self::NativeMedia => "native_media",
            Self::SystemMetadata => "system_metadata",
        }
    }

    fn grounding_role(self) -> GroundingRole {
        match self {
            Self::Statement | Self::Event | Self::NativeMedia => GroundingRole::Original,
            Self::SessionDate => GroundingRole::Context,
            Self::Entity | Self::SystemMetadata => GroundingRole::ExpansionOnly,
        }
    }
}

#[derive(Debug, Clone)]
struct EvidenceAtom {
    atom_id: String,
    memory_id: u64,
    kind: EvidenceKind,
    modality: String,
    text: String,
    source: String,
    source_anchor: Option<AnswerEvidenceSourceAnchor>,
    speaker: Option<String>,
    entity: Option<String>,
    section_scope: Vec<String>,
    /// The person or people the statement is actually about. This is kept
    /// separate from the turn speaker and from names merely addressed.
    fact_subjects: Vec<String>,
    /// Named people mentioned as participants or objects of a relation. They
    /// are useful retrieval constraints, but cannot satisfy a subject slot.
    mentioned_people: Vec<String>,
    /// Names used as a direct address, such as `Thanks, Caroline`.
    addressed_people: Vec<String>,
    /// Explicit possessive owners, such as `Caroline's plan`.
    possessive_owners: Vec<String>,
    /// Reproducible explanation for fact-subject attribution.
    subject_binding: Vec<String>,
    base_score: f32,
    score: f32,
    supports_slots: Vec<String>,
    context_group: usize,
    ordinal: usize,
    /// Reranker probability that this atom's turn answers the question;
    /// `None` when no semantic scorer ran.
    semantic_relevance: Option<f32>,
}

#[derive(Debug, Clone, Default)]
struct FactAttribution {
    fact_subjects: Vec<String>,
    mentioned_people: Vec<String>,
    addressed_people: Vec<String>,
    possessive_owners: Vec<String>,
    subject_binding: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cardinality {
    Single,
    List,
    Count,
}

impl Cardinality {
    fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::List => "list",
            Self::Count => "count",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerTarget {
    Temporal,
    Location,
    Count,
    Duration,
    List,
    Reason,
    Initiation,
    Manner,
    Description,
    Event,
    Entity,
    Object,
}

impl AnswerTarget {
    fn as_str(self) -> &'static str {
        match self {
            Self::Temporal => "temporal",
            Self::Location => "location",
            Self::Count => "count",
            Self::Duration => "duration",
            Self::List => "list",
            Self::Reason => "reason",
            Self::Initiation => "initiation",
            Self::Manner => "manner",
            Self::Description => "description",
            Self::Event => "event",
            Self::Entity => "entity",
            Self::Object => "object",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AnswerShape {
    Scalar,
    AspectSet,
    ActionSet,
    List,
    BoundedList,
    CountAggregate,
    Temporal,
    Location,
    Initiation,
    Likely,
    Counterfactual,
}

impl AnswerShape {
    fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::AspectSet => "aspect_set",
            Self::ActionSet => "action_set",
            Self::List => "list",
            Self::BoundedList => "bounded_list",
            Self::CountAggregate => "count_aggregate",
            Self::Temporal => "event_time",
            Self::Location => "location",
            Self::Initiation => "initiation",
            Self::Likely => "likely",
            Self::Counterfactual => "counterfactual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InferenceType {
    Direct,
    Likely,
    Counterfactual,
}

/// The direction of a condition matters to a final Claim. `unless` and
/// `only when` cannot be treated as interchangeable keyword matches for
/// `if`, even when they mention the same nouns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConditionRelation {
    None,
    If,
    When,
    OnlyAfter,
    OnlyBefore,
    Unless,
    OnlyWhen,
    Assuming,
    Causal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnswerOperator {
    NovelChapterRangeFromRecords,
    NovelChapterFromRecords,
    OperationalHabit,
    VerifiedHandoff,
    OperatorBriefing,
    AmbientCapture,
    LearningResume,
    DocumentComparison,
    ExperienceRecall,
    UniversalProfile,
    DesignStyle,
    CategoricalAttribute,
    SharedLocation,
    MultimodalAppearanceChange,
    RecentVisualCreation,
    GroundedList,
    GroundedInference,
    ResearchObject,
    LinkedOrigin,
    EventDate,
    TemporalReasoning,
    CrossSourceEventCount,
    FocusedScalar,
    NovelCrossChapter,
    NovelChapter,
    NovelChapterUnavailable,
    NovelStyle,
    UnifiedGeneral,
    AnswerabilityGate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnswerOperatorContract {
    CandidateClaims,
    CheckedPresentation,
    CoverageFeedback,
}

impl AnswerOperator {
    pub(crate) fn contract(self) -> AnswerOperatorContract {
        match self {
            Self::NovelChapterRangeFromRecords
            | Self::NovelChapterFromRecords
            | Self::OperationalHabit
            | Self::VerifiedHandoff
            | Self::OperatorBriefing
            | Self::AmbientCapture
            | Self::LearningResume
            | Self::DocumentComparison
            | Self::ExperienceRecall
            | Self::UniversalProfile
            | Self::DesignStyle
            | Self::CategoricalAttribute
            | Self::SharedLocation
            | Self::MultimodalAppearanceChange
            | Self::RecentVisualCreation
            | Self::GroundedList
            | Self::GroundedInference
            | Self::ResearchObject
            | Self::LinkedOrigin
            | Self::EventDate
            | Self::TemporalReasoning
            | Self::CrossSourceEventCount
            | Self::FocusedScalar
            | Self::NovelCrossChapter
            | Self::NovelChapter
            | Self::NovelStyle => AnswerOperatorContract::CandidateClaims,
            Self::UnifiedGeneral => AnswerOperatorContract::CheckedPresentation,
            Self::NovelChapterUnavailable | Self::AnswerabilityGate => {
                AnswerOperatorContract::CoverageFeedback
            }
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::NovelChapterRangeFromRecords => "novel_chapter_range_records_v1",
            Self::NovelChapterFromRecords => "novel_chapter_records_v1",
            Self::OperationalHabit => "operational_habit_v1",
            Self::VerifiedHandoff => "verified_handoff_v1",
            Self::OperatorBriefing => "operator_briefing_v1",
            Self::AmbientCapture => "ambient_capture_v1",
            Self::LearningResume => "learning_resume_v1",
            Self::DocumentComparison => "document_comparison_v1",
            Self::ExperienceRecall => "experience_recall_v1",
            Self::UniversalProfile => "universal_profile_v1",
            Self::DesignStyle => "design_style_v1",
            Self::CategoricalAttribute => "categorical_attribute_v1",
            Self::SharedLocation => "shared_location_v1",
            Self::MultimodalAppearanceChange => "multimodal_appearance_change_v1",
            Self::RecentVisualCreation => "recent_visual_creation_v1",
            Self::GroundedList => "grounded_list_v1",
            Self::GroundedInference => "grounded_inference_v1",
            Self::ResearchObject => "research_object_v1",
            Self::LinkedOrigin => "linked_origin_v1",
            Self::EventDate => "event_date_v1",
            Self::TemporalReasoning => "temporal_reasoning_v1",
            Self::CrossSourceEventCount => "cross_source_event_count_v1",
            Self::FocusedScalar => "focused_scalar_v1",
            Self::NovelCrossChapter => "novel_cross_chapter_v1",
            Self::NovelChapter => "novel_chapter_v1",
            Self::NovelChapterUnavailable => "novel_chapter_unavailable_v1",
            Self::NovelStyle => "novel_style_v1",
            Self::UnifiedGeneral => "unified_answer_generator_v1",
            Self::AnswerabilityGate => "answerability_gate_v1",
        }
    }
}

impl InferenceType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Likely => "likely",
            Self::Counterfactual => "counterfactual",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TemporalGranularity {
    Unspecified,
    Year,
    Month,
    Week,
    Day,
}

impl TemporalGranularity {
    fn as_str(self) -> &'static str {
        match self {
            Self::Unspecified => "unspecified",
            Self::Year => "year",
            Self::Month => "month",
            Self::Week => "week",
            Self::Day => "day",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum TemporalConstraint {
    None,
    MonthOfYear {
        month: u32,
        year: Option<i32>,
    },
    WeekOfMonth {
        week: u8,
        month: u32,
        year: Option<i32>,
    },
    WeekBeforeDate(chrono::NaiveDate),
    WeekAfterDate(chrono::NaiveDate),
}

impl TemporalConstraint {
    fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }

    fn granularity(&self) -> Option<TemporalGranularity> {
        match self {
            Self::None => None,
            Self::MonthOfYear { .. } => Some(TemporalGranularity::Month),
            Self::WeekOfMonth { .. } | Self::WeekBeforeDate(_) | Self::WeekAfterDate(_) => {
                Some(TemporalGranularity::Week)
            }
        }
    }

    fn as_str(&self) -> String {
        match self {
            Self::None => "none".to_string(),
            Self::MonthOfYear { month, year } => year
                .map(|year| format!("month_of_year:{year}-{month:02}"))
                .unwrap_or_else(|| format!("month_of_year:{month:02}")),
            Self::WeekOfMonth { week, month, year } => year
                .map(|year| format!("week_of_month:{year}-{month:02}:{week}"))
                .unwrap_or_else(|| format!("week_of_month:{month:02}:{week}")),
            Self::WeekBeforeDate(date) => format!("week_before:{}", date.format("%F")),
            Self::WeekAfterDate(date) => format!("week_after:{}", date.format("%F")),
        }
    }

    fn control_terms(&self) -> Vec<&'static str> {
        match self {
            Self::None => Vec::new(),
            Self::MonthOfYear { month, .. } => vec![temporal_month_name(*month)],
            Self::WeekOfMonth { week, .. } => match week {
                1 => vec!["first", "1st", "week"],
                2 => vec!["second", "2nd", "week"],
                3 => vec!["third", "3rd", "week"],
                4 => vec!["fourth", "4th", "week"],
                5 => vec!["fifth", "5th", "week"],
                _ => vec!["week"],
            },
            Self::WeekBeforeDate(_) => vec!["week", "before"],
            Self::WeekAfterDate(_) => vec!["week", "after"],
        }
    }

    fn matches(&self, date: chrono::NaiveDate) -> bool {
        match self {
            Self::None => true,
            Self::MonthOfYear { month, year } => {
                date.month() == *month && year.is_none_or(|year| date.year() == year)
            }
            Self::WeekOfMonth { week, month, year } => {
                date.month() == *month
                    && year.is_none_or(|year| date.year() == year)
                    && ((date.day0() / 7) + 1) == u32::from(*week)
            }
            Self::WeekBeforeDate(anchor) => {
                let start = anchor
                    .checked_sub_signed(chrono::Duration::days(7))
                    .unwrap_or(*anchor);
                date >= start && date < *anchor
            }
            Self::WeekAfterDate(anchor) => {
                let end = anchor
                    .checked_add_signed(chrono::Duration::days(7))
                    .unwrap_or(*anchor);
                date > *anchor && date <= end
            }
        }
    }
}

#[derive(Debug, Clone)]
struct QueryPlan {
    subject: Option<String>,
    subject_is_action_agent: bool,
    people: Vec<String>,
    named_entities: Vec<String>,
    answer_target: AnswerTarget,
    answer_shape: AnswerShape,
    categorical_attribute_kind: Option<crate::memory_engine_summary::CategoricalAnswerKind>,
    grounded_list_kind: Option<crate::memory_engine_summary::GroundedListKind>,
    list_requires_household_context: bool,
    household_dependent_preference: bool,
    requires_complementary_action: bool,
    achievement_extremum: bool,
    temporal_scope: String,
    media_provenance_requested: bool,
    co_participant_event_requested: bool,
    recommendation_requested: bool,
    recommendation_topic_terms: Vec<String>,
    temporal_granularity: TemporalGranularity,
    temporal_constraint: TemporalConstraint,
    duration_clauses: Option<(String, String)>,
    time_terms: Vec<String>,
    participant_relationship_qualifier: bool,
    structural_selectors: Vec<String>,
    action_terms: Vec<String>,
    object_terms: Vec<String>,
    answer_class_terms: Vec<String>,
    qualifier_terms: Vec<String>,
    cardinality: Cardinality,
    quantity_terms: Vec<String>,
    list_scope: String,
    inference_type: InferenceType,
    negation_terms: Vec<String>,
    condition_terms: Vec<String>,
    condition_relation: ConditionRelation,
    protected_terms: Vec<String>,
    requested_modalities: Vec<String>,
    required_slots: Vec<String>,
    query_terms: Vec<String>,
    /// Relation nouns the question names as a possessed actor
    /// ("Jolene's parents"), distinct from coordinated participants.
    possessive_relation_actors: Vec<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct UnifiedAnswerDraft {
    pub(crate) text: String,
    pub(crate) draft_mode: &'static str,
    rendered_claims: Vec<AnswerClaimPlanReceipt>,
}

/// The only evidence shape that may graduate into a final answer Claim.
/// Record IDs remain for backwards compatibility, but are derived from the
/// exact atoms below and can never stand in for them.
#[derive(Debug, Clone, Default)]
pub(crate) struct VerifiedClaimEvidence {
    pub(crate) memory_ids: Vec<u64>,
    pub(crate) atom_ids: Vec<String>,
    pub(crate) source_anchors: Vec<AnswerEvidenceSourceAnchor>,
    pub(crate) derivation: Vec<String>,
}

impl VerifiedClaimEvidence {
    pub(crate) fn is_bound(&self) -> bool {
        !self.atom_ids.is_empty() && !self.memory_ids.is_empty()
    }
}

/// A single, reproducible count proof shared by rendering and final Claim
/// verification. UI node limits intentionally do not participate here: a
/// concise answer may still require many atoms to prove its number.
#[derive(Debug, Clone, Default)]
struct CountEventSet {
    atom_ids: Vec<String>,
    memory_ids: Vec<u64>,
    numeric_value: usize,
    display_value: String,
    derivation: Vec<String>,
}

impl CountEventSet {
    fn is_proven(&self) -> bool {
        !self.atom_ids.is_empty() && !self.memory_ids.is_empty()
    }
}

#[derive(Debug, Clone)]
pub(crate) struct PlannedAnswerPipeline {
    pub(crate) receipt: AnswerPipelineReceipt,
    pub(crate) unified_draft: Option<UnifiedAnswerDraft>,
    query_plan: QueryPlan,
    operators: Vec<AnswerOperator>,
    evidence_atoms: Vec<EvidenceAtom>,
    source_fingerprints: HashMap<u64, String>,
    rejected_source_ids: HashSet<u64>,
    /// Set when the selected renderer answers a general memory question;
    /// specialized renderers (chapters, comparisons, briefings) own their
    /// relevance and are never gated by question relevance.
    general_question_operator: bool,
}

/// Immutable source identity shared by live model reasoning and persisted
/// evidence projections. Scores are deliberately excluded: retrieval order may
/// change without changing the source whose spans are being bound.
pub(crate) fn reasoning_source_fingerprint(item: &RetrievedItem) -> String {
    let input = serde_json::to_vec(&(
        &item.content,
        &item.source,
        &item.entity,
        item.version,
        item.event_metadata.is_some() || item.content.trim_start().starts_with("event:"),
    ))
    .expect("projection input contains only serializable primitives");
    crate::shared_object::stable_digest(&input)
}

/// Call only with the authorized retrieval snapshot, before answer selection.
/// This projection deliberately does not run the local answer-bearing filter.
pub(crate) fn reasoning_source_atoms(
    items: &[RetrievedItem],
) -> Result<BTreeMap<String, crate::answer_reasoning_contract::SourceAtom>, String> {
    use crate::answer_reasoning_contract::SourceAtom;
    let mut fingerprints = HashMap::new();
    let mut output = BTreeMap::new();
    for item in items {
        // A derived person profile is not evidence: nobody said it. It reaches
        // the model as a separate, labelled section (see `person_profile`).
        if crate::person_profile::is_person_profile_record(&item.memory_type, &item.content) {
            continue;
        }
        let digest = reasoning_source_fingerprint(item);
        if let Some(previous) = fingerprints.insert(item.id, digest.clone()) {
            if previous != digest {
                return Err("source_snapshot_changed".into());
            }
            continue;
        }
        for atom in project_source(ProjectionInput {
            memory_id: item.id,
            content: &item.content,
            source: &item.source,
            entity: item.entity.as_deref(),
            score: item.score,
            is_event: item.event_metadata.is_some()
                || item.content.trim_start().starts_with("event:"),
        }) {
            let source = SourceAtom {
                atom_id: atom.atom_id.clone(),
                memory_id: item.id,
                source_version: item.version,
                source_digest: digest.clone(),
                body: atom.text,
                modality: atom.modality,
                evidence_kind: atom.kind.as_str().into(),
                speaker: atom.speaker,
                context_group: atom.context_group,
                descriptive_only: atom.kind == EvidenceKind::NativeMedia,
                source_anchor: atom.source_anchor,
            };
            if output.insert(atom.atom_id, source).is_some() {
                return Err("duplicate_source_atom".into());
            }
        }
    }
    Ok(output)
}

impl PlannedAnswerPipeline {
    fn refresh_structural_coverage(&mut self) {
        let updated = assess_slot_coverage(&self.query_plan, &self.evidence_atoms, 0, 0);
        for replacement in updated.into_iter().filter(|slot|
            slot.slot == "structural_selector" || slot.slot.starts_with("structural_selector:")) {
            if let Some(current) = self.receipt.slot_coverage.iter_mut().find(|slot| slot.slot == replacement.slot) {
                *current = replacement;
            }
        }
    }

    /// Authorization is checked by the caller. Admission adds candidates, not
    /// proof, and cannot redefine a frozen aggregate after generation.
    pub(crate) fn admit_renderer_evidence(&mut self, items: &[RetrievedItem]) {
        let mut conflicts = BTreeSet::new();
        for item in items {
            let fingerprint = reasoning_source_fingerprint(item);
            let expected = self
                .source_fingerprints
                .entry(item.id)
                .or_insert_with(|| fingerprint.clone());
            if *expected != fingerprint && self.rejected_source_ids.insert(item.id) {
                conflicts.insert(item.id);
            }
        }
        if !conflicts.is_empty() {
            self.evidence_atoms
                .retain(|atom| !self.rejected_source_ids.contains(&atom.memory_id));
            let evidence_ids = conflicts.into_iter().collect::<Vec<_>>();
            self.receipt.slot_coverage.push(AnswerSlotCoverageReceipt {
                slot: "source_snapshot".into(),
                status: "partial".into(),
                evidence_ids: evidence_ids.clone(),
            });
            self.receipt.stage_trace.push(AnswerPipelineStageReceipt {
                stage: "renderer_evidence_admission".into(),
                status: "failed".into(),
                input_count: evidence_ids.len(),
                output_count: 0,
                evidence_ids,
                reason: Some(
                    "source_snapshot_changed: restart planning; source versions cannot be mixed"
                        .into(),
                ),
            });
            self.receipt
                .evidence_plan
                .selected_atoms
                .retain(|atom| !self.rejected_source_ids.contains(&atom.memory_id));
            self.receipt.evidence_plan.selected_atom_count = self.evidence_atoms.len();
            self.refresh_structural_coverage();
        }
        if self.query_plan.answer_target == AnswerTarget::Count {
            return;
        }
        let mut existing = self
            .evidence_atoms
            .iter()
            .map(|atom| atom.atom_id.clone())
            .collect::<HashSet<_>>();
        let mut projected_sources = HashSet::new();
        let mut added = items
            .iter()
            .filter(|item| {
                !self.rejected_source_ids.contains(&item.id) && projected_sources.insert(item.id)
            })
            .flat_map(|item| {
                project_source(ProjectionInput {
                    memory_id: item.id,
                    content: &item.content,
                    source: &item.source,
                    entity: item.entity.as_deref(),
                    score: item.score,
                    is_event: item.event_metadata.is_some()
                        || item.content.trim_start().starts_with("event:"),
                })
            })
            .collect::<Vec<_>>();
        if added.is_empty() {
            return;
        }
        rank_evidence_atoms(&self.query_plan, &mut added);
        annotate_evidence_chain_candidates(&self.query_plan, &mut added);
        // Rank with the full source context before discarding already admitted atoms.
        added.retain(|atom| existing.insert(atom.atom_id.clone()));
        if added.is_empty() {
            return;
        }
        added.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.memory_id.cmp(&right.memory_id))
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        });
        let input_count = added.len();
        let remaining = MAX_RUNTIME_ATOMS.saturating_sub(self.evidence_atoms.len());
        added.truncate(remaining);
        let admitted_ids = added
            .iter()
            .map(|atom| atom.atom_id.as_str())
            .collect::<HashSet<_>>();
        self.receipt
            .evidence_plan
            .rejected_atoms
            .retain(|atom| !admitted_ids.contains(atom.atom_id.as_str()));
        let ids = added
            .iter()
            .map(|atom| atom.memory_id)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        for atom in &added {
            if self.receipt.evidence_plan.selected_atoms.len() < MAX_RECEIPT_ATOMS {
                self.receipt
                    .evidence_plan
                    .selected_atoms
                    .push(evidence_atom_receipt(atom));
            }
        }
        self.receipt.stage_trace.push(AnswerPipelineStageReceipt {
            stage: "renderer_evidence_admission".into(),
            status: if input_count > remaining { "partial" } else { "completed" }.into(),
            input_count, output_count: added.len(), evidence_ids: ids.clone(),
            reason: Some("authorized candidates projected under the unchanged question plan; final proof required".into()),
        });
        if input_count > remaining {
            self.receipt.slot_coverage.push(AnswerSlotCoverageReceipt {
                slot: "proof_scope".into(),
                status: "partial".into(),
                evidence_ids: ids,
            });
        }
        self.evidence_atoms.extend(added);
        self.receipt.evidence_plan.selected_atom_count = self.evidence_atoms.len();
        self.refresh_structural_coverage();
    }

    pub(crate) fn operators(&self) -> &[AnswerOperator] {
        &self.operators
    }
    

    pub(crate) fn selected_memory_ids(&self) -> &[u64] {
        &self.receipt.evidence_plan.selected_memory_ids
    }

    /// Records the renderer chosen for this answer; general memory-question
    /// operators enable the semantic relevance gate.
    pub(crate) fn set_selected_operator(&mut self, operator: Option<&str>) {
        self.general_question_operator = matches!(
            operator,
            Some(
                "unified_answer_generator_v1"
                    | "focused_scalar_v1"
                    | "grounded_list_v1"
                    | "grounded_inference_v1"
                    | "shared_location_v1"
            )
        );
    }

    /// For general memory questions, a Claim whose cited sentences were
    /// reranked for this question must be confidently relevant. Measured on
    /// the LoCoMo dev split, lexical matching in the uncertain band verified
    /// wrong answers 87% of the time for non-temporal Claims. Dates, counts and
    /// durations keep their arithmetic proofs; typed attribute values keep
    /// their rule extraction.
    fn semantic_relevance_gate_rejects(&self, atom_ids: &[String], derivation: &[String]) -> bool {
        if !self.general_question_operator
            || atom_ids.is_empty()
            || matches!(
                self.query_plan.answer_target,
                AnswerTarget::Temporal | AnswerTarget::Count | AnswerTarget::Duration
            )
            || derivation.iter().any(|step| {
                step.contains("date") || step.starts_with("count_") || step.starts_with("duration")
            })
        {
            return false;
        }
        let cited = self
            .evidence_atoms
            .iter()
            .filter(|atom| {
                atom_ids.contains(&atom.atom_id)
                    && atom.kind.grounding_role() == GroundingRole::Original
            })
            .collect::<Vec<_>>();
        if cited
            .iter()
            .any(|atom| atom.supports_slots.iter().any(|slot| slot == "categorical_attribute_value"))
        {
            return false;
        }
        cited
            .iter()
            .filter_map(|atom| atom.semantic_relevance)
            .reduce(f32::max)
            .is_some_and(|best| best < crate::semantic_scorer::grant_threshold())
    }

    pub(crate) fn verify_claim_evidence(
        &self,
        claim: &str,
        candidate_ids: &HashSet<u64>,
    ) -> VerifiedClaimEvidence {
        if (candidate_ids.is_empty() && !self.rejected_source_ids.is_empty())
            || candidate_ids
                .iter()
                .any(|id| self.rejected_source_ids.contains(id))
        {
            return VerifiedClaimEvidence::default();
        }
        if self.query_plan.duration_clauses.is_some()
            && self
                .receipt
                .slot_coverage
                .iter()
                .any(|slot| slot.slot == "proof_scope" && slot.status != "covered")
        {
            return VerifiedClaimEvidence::default();
        }
        // A record is only a retrieval container. Final support must be proved
        // from exact original atoms, never from a record-level lexical match.
        let verified_memory_ids = if candidate_ids.is_empty() {
            self.evidence_atoms
                .iter()
                .map(|atom| atom.memory_id)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>()
        } else {
            candidate_ids.iter().copied().collect::<Vec<_>>()
        };
        let (atom_ids, source_anchors, derivation) = verified_claim_bindings_for_plan(
            &self.query_plan,
            &self.evidence_atoms,
            claim,
            &verified_memory_ids,
        );
        if semantic_contradiction_veto(&self.query_plan, &self.evidence_atoms, &atom_ids, claim)
            || self.semantic_relevance_gate_rejects(&atom_ids, &derivation)
        {
            return VerifiedClaimEvidence::default();
        }
        let memory_ids = selected_memory_ids_for_atoms(&atom_ids, &self.evidence_atoms);
        VerifiedClaimEvidence {
            memory_ids,
            atom_ids,
            source_anchors,
            derivation,
        }
    }

    /// Binds a specialized renderer's Claim that is a verbatim quotation of
    /// one cited record. The quotation may span several consecutive atoms (the
    /// projector splits sentences, clauses, and fields). The renderer selected
    /// the record, so question relevance is its responsibility; support is
    /// proved by exact containment plus the shared negation/condition logic
    /// check against the quoted source span. Short fragments (fewer than three
    /// informative terms) never qualify.
    pub(crate) fn verify_verbatim_renderer_quotation(
        &self,
        claim: &str,
        candidate_ids: &HashSet<u64>,
    ) -> VerifiedClaimEvidence {
        if candidate_ids.is_empty()
            || candidate_ids
                .iter()
                .any(|id| self.rejected_source_ids.contains(id))
        {
            return VerifiedClaimEvidence::default();
        }
        let claim_text = citation_regex().replace_all(claim, " ");
        let claim_text = clean_answer_statement(&claim_text);
        if informative_terms(&claim_text).len() < 3 {
            return VerifiedClaimEvidence::default();
        }
        let needle = normalize_for_dedup(&claim_text);
        let claim_lower = claim_text.to_lowercase();
        // The projector lifts a leading `Label:` into `speaker`; the source
        // line is the label plus the atom text.
        let source_line = |atom: &EvidenceAtom| {
            atom.speaker
                .as_deref()
                .map(|speaker| format!("{speaker}: {}", atom.text))
                .unwrap_or_else(|| atom.text.clone())
        };
        let mut candidates = candidate_ids.iter().copied().collect::<Vec<_>>();
        candidates.sort_unstable();
        for memory_id in candidates {
            let mut record_atoms = self
                .evidence_atoms
                .iter()
                .filter(|atom| {
                    atom.memory_id == memory_id
                        && atom.kind.grounding_role() == GroundingRole::Original
                })
                .collect::<Vec<_>>();
            record_atoms.sort_by_key(|atom| atom.ordinal);
            record_atoms.dedup_by(|left, right| left.atom_id == right.atom_id);
            // Smallest run of consecutive atoms whose joined text contains the
            // quotation. A lifted label can only prefix the first atom of the
            // run; inner atoms join on their own text.
            let raw = record_atoms
                .iter()
                .map(|atom| normalize_for_dedup(&atom.text))
                .collect::<Vec<_>>();
            let led = record_atoms
                .iter()
                .map(|atom| normalize_for_dedup(&source_line(atom)))
                .collect::<Vec<_>>();
            let run_contains = |start: usize, end: usize| {
                let tail = raw[start + 1..=end].concat();
                format!("{}{tail}", raw[start]).contains(&needle)
                    || format!("{}{tail}", led[start]).contains(&needle)
            };
            // Prefer an atom that is exactly the quotation over any wider run.
            let exact_atom = (0..record_atoms.len())
                .find(|index| raw[*index] == needle || led[*index] == needle);
            let span = exact_atom.map(|index| (index, index)).or_else(|| (0..record_atoms.len()).find_map(|start| {
                let mut length = led[start].len();
                for end in start..record_atoms.len() {
                    if end > start {
                        length += raw[end].len();
                    }
                    if run_contains(start, end) {
                        return Some((start, end));
                    }
                    if length > needle.len() + led[start].len() {
                        break;
                    }
                }
                None
            }));
            let Some((mut start, end)) = span else {
                continue;
            };
            // Minimal span: drop leading atoms the quotation does not need, so
            // unrelated neighbouring text never enters the logic check.
            while start < end && run_contains(start + 1, end) {
                start += 1;
            }
            let quoted = &record_atoms[start..=end];
            let source_text = quoted
                .iter()
                .enumerate()
                .map(|(index, atom)| if index == 0 { source_line(atom) } else { atom.text.clone() })
                .collect::<Vec<_>>()
                .join(" ")
                .to_lowercase();
            // An exact quotation reproduces the source's own negations and
            // conditions; only a partial quotation can drop or flip them.
            let exact_quotation = {
                let tail = raw[start + 1..=end].concat();
                format!("{}{tail}", raw[start]) == needle
                    || format!("{}{tail}", led[start]) == needle
            };
            if !exact_quotation
                && !claim_logic_matches_source(&self.query_plan, &claim_lower, &source_text)
            {
                continue;
            }
            let (atom_ids, source_anchors, mut derivation) = claim_evidence_binding_fields(quoted);
            if self.semantic_relevance_gate_rejects(&atom_ids, &derivation) {
                continue;
            }
            derivation.push("verbatim_renderer_quotation_v1".into());
            return VerifiedClaimEvidence {
                memory_ids: selected_memory_ids_for_atoms(&atom_ids, &self.evidence_atoms),
                atom_ids,
                source_anchors,
                derivation,
            };
        }
        VerifiedClaimEvidence::default()
    }

    

    /// Returns verified coverage for list claims emitted by the unified renderer.
    pub(crate) fn unified_draft_coverage(&self) -> Option<Vec<CoverageReport>> {
        let draft = self.unified_draft.as_ref()?;
        (!draft.rendered_claims.is_empty()).then(|| {
            draft
                .rendered_claims
                .iter()
                .map(|claim| {
                    let text = claim.text.clone().unwrap_or_default();
                    let candidates = claim.evidence_ids.iter().copied().collect::<HashSet<_>>();
                    let inferred_recommendation = claim.claim_type == "inference"
                        && self.receipt.query_plan.recommendation_requested
                        && ["recommendation_capability", "recommendation_topic"]
                            .iter()
                            .all(|slot| {
                                self.evidence_atoms.iter().any(|atom| {
                                    candidates.contains(&atom.memory_id)
                                        && atom.supports_slots.iter().any(|value| value == slot)
                                })
                            });
                    let proof = if text.is_empty()
                        || candidates.is_empty()
                        || candidates
                            .iter()
                            .any(|id| self.rejected_source_ids.contains(id))
                    {
                        VerifiedClaimEvidence::default()
                    } else if inferred_recommendation {
                        let (atom_ids, source_anchors, derivation) =
                            claim_evidence_bindings_for_atom_ids(
                                &self.evidence_atoms,
                                &claim.evidence_atom_ids,
                            );
                        let memory_ids =
                            selected_memory_ids_for_atoms(&atom_ids, &self.evidence_atoms);
                        VerifiedClaimEvidence {
                            memory_ids,
                            atom_ids,
                            source_anchors,
                            derivation,
                        }
                    } else {
                        self.verify_claim_evidence(&text, &candidates)
                    };
                    // A renderer's planned atom list is useful provenance, but it is
                    // not itself proof. Direct claims must re-bind their exact atoms
                    // against the final text after the relation validator succeeds.
                    let is_bound = proof.is_bound();
                    let verification_status = if inferred_recommendation && is_bound {
                        "inferred".to_string()
                    } else if !is_bound {
                        "unsupported".to_string()
                    } else {
                        "verified".to_string()
                    };
                    CoverageReport {
                        claim: text,
                        covered: is_bound,
                        evidence_ids: proof.memory_ids,
                        evidence_atom_ids: proof.atom_ids,
                        evidence_source_anchors: proof.source_anchors,
                        derivation: proof.derivation,
                        verification_status,
                    }
                })
                .collect()
        })
    }

    /// Normal renderers consume atom-level evidence so unrelated sentences in
    /// the same long record cannot displace selected facts.
    pub(crate) fn scoped_items(&self, items: &[RetrievedItem]) -> Vec<RetrievedItem> {
        let selected_atom_ids = self
            .evidence_atoms
            .iter()
            .filter(|atom| {
                matches!(
                    atom.kind.grounding_role(),
                    GroundingRole::Original | GroundingRole::Context
                )
            })
            .map(|atom| atom.atom_id.clone())
            .collect::<HashSet<_>>();
        self.scoped_items_for_atom_ids(items, &selected_atom_ids)
    }

    /// A list value must originate in a selected primary atom. Supporting
    /// context may constrain a list, but cannot become an additional value.
    /// An unnamed book reference is retained so the renderer can report a
    /// partial answer rather than inventing a title.
    pub(crate) fn grounded_list_items(&self, items: &[RetrievedItem]) -> Vec<RetrievedItem> {
        let selected_answer_atom_ids = self
            .receipt
            .evidence_plan
            .selected_answer_atom_ids
            .iter()
            .map(String::as_str)
            .collect::<HashSet<_>>();
        let same_turn_list_bridge_contexts = self
            .evidence_atoms
            .iter()
            .filter(|atom| {
                selected_answer_atom_ids.contains(atom.atom_id.as_str())
                    && atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "same_turn_list_value_bridge")
            })
            .map(|atom| (atom.memory_id, atom.context_group))
            .collect::<HashSet<_>>();
        let selected_atom_ids = self
            .evidence_atoms
            .iter()
            .filter(|atom| {
                atom.kind.grounding_role() == GroundingRole::Original
                    && (selected_answer_atom_ids.contains(atom.atom_id.as_str())
                        || atom
                            .supports_slots
                            .iter()
                            .any(|slot| slot == "list_unresolved_reference")
                        || (same_turn_list_bridge_contexts
                            .contains(&(atom.memory_id, atom.context_group))
                            && crate::memory_engine_summary::source_signals_decompression(
                                &atom.text,
                            )))
            })
            .map(|atom| atom.atom_id.clone())
            .collect::<HashSet<_>>();
        self.scoped_items_for_atom_ids(items, &selected_atom_ids)
    }

    /// The record scope needed for rendering includes explicit non-answer
    /// context such as an unnamed book reference, while preserving answer
    /// atoms before contextual additions.
    pub(crate) fn render_memory_ids(&self) -> Vec<u64> {
        let mut memory_ids = self.selected_memory_ids().to_vec();
        for atom in &self.evidence_atoms {
            if !memory_ids.contains(&atom.memory_id) {
                memory_ids.push(atom.memory_id);
            }
        }
        memory_ids
    }

    fn scoped_items_for_atom_ids(
        &self,
        items: &[RetrievedItem],
        selected_atom_ids: &HashSet<String>,
    ) -> Vec<RetrievedItem> {
        let mut selected_by_memory = HashMap::<u64, Vec<&EvidenceAtom>>::new();
        for atom in &self.evidence_atoms {
            if selected_atom_ids.contains(&atom.atom_id) {
                selected_by_memory
                    .entry(atom.memory_id)
                    .or_default()
                    .push(atom);
            }
        }

        items
            .iter()
            .filter_map(|item| {
                let mut atoms = selected_by_memory.remove(&item.id)?;
                atoms.sort_by(|left, right| left.ordinal.cmp(&right.ordinal));
                let content = atoms
                    .iter()
                    .map(|atom| scoped_atom_text(atom))
                    .collect::<Vec<_>>()
                    .join("\n");
                (!content.trim().is_empty()).then(|| {
                    let mut scoped = item.clone();
                    scoped.content = content;
                    scoped
                })
            })
            .collect()
    }

    pub(crate) fn operator_is_admissible(&self, operator: AnswerOperator) -> bool {
        let status = |slot: &str| {
            self.receipt
                .slot_coverage
                .iter()
                .find(|coverage| coverage.slot == slot)
                .map(|coverage| coverage.status.as_str())
                .unwrap_or("covered")
        };
        let target_available = match self.receipt.query_plan.answer_shape.as_str() {
            "location" => status("location") != "missing",
            "event_time" => status("event_time") != "missing",
            _ => true,
        };
        let core_available = status("answer_bearing_fact") != "missing"
            && status("subject") != "missing"
            && status("action") != "missing"
            && status("action_object_complement") != "missing"
            && status("observed_action_fact") != "missing"
            && status("categorical_attribute_value") != "missing"
            && status("duration_value") != "missing"
            && status("list_value") != "missing"
            && status("list_participant_scope") != "missing"
            && status("achievement_extremum") != "missing"
            && status("answer_class") != "missing"
            && status("qualifier_context") != "missing";
        let explicit_attribution_conflict =
            self.receipt
                .evidence_plan
                .selected_atoms
                .iter()
                .any(|atom| {
                    atom.supports_slots.iter().any(|slot| {
                        matches!(
                            slot.as_str(),
                            "relationship_actor_conflict" | "collective_participant_context_bridge"
                        )
                    }) && atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "answer_bearing_fact")
                })
                && status("subject") == "missing"
                && status("qualifier_context") != "missing";
        let multi_entity_inference = self.receipt.query_plan.inference_type == "likely"
            && self.receipt.query_plan.named_entities.len() > 1;
        let entity_context_partial = multi_entity_inference
            && self
                .unified_draft
                .as_ref()
                .is_some_and(|draft| draft.draft_mode == "partial")
            && self
                .evidence_atoms
                .iter()
                .any(atom_is_inference_entity_context);

        match operator {
            AnswerOperator::EventDate => {
                core_available
                    && status("temporal_scope") == "covered"
                    && status("event_time") == "covered"
            }
            AnswerOperator::TemporalReasoning => {
                core_available
                    && status("temporal_scope") != "missing"
                    && status("event_time") != "missing"
            }
            AnswerOperator::CrossSourceEventCount => {
                core_available && status("quantity") != "missing"
            }
            AnswerOperator::GroundedList => {
                !multi_entity_inference && core_available && status("list_scope") != "missing"
            }
            AnswerOperator::GroundedInference => {
                !multi_entity_inference && core_available && status("inference_basis") != "missing"
            }
            AnswerOperator::UnifiedGeneral => {
                entity_context_partial
                    || (target_available && (core_available || explicit_attribution_conflict))
            }
            _ => true,
        }
    }

    
    
}





#[derive(Debug, Clone, Serialize)]
pub(crate) struct PersistedEvidenceAtom {
    pub(crate) memory_id: i64,
    pub(crate) conversation_id: String,
    pub(crate) atom_id: String,
    pub(crate) atom_kind: String,
    pub(crate) modality: String,
    pub(crate) ordinal: i32,
    pub(crate) speaker: Option<String>,
    pub(crate) entity: Option<String>,
    pub(crate) body: String,
    pub(crate) source: String,
    pub(crate) source_version: i32,
    pub(crate) content_digest: String,
    pub(crate) metadata: serde_json::Value,
}

struct ProjectionInput<'a> {
    memory_id: u64,
    content: &'a str,
    source: &'a str,
    entity: Option<&'a str>,
    score: f32,
    is_event: bool,
}

/// Agent-turn endpoints prepend recalled Memory Files / procedure context to
/// the task. That context is evidence for renderers, not part of the question:
/// planning over it turns file headings, SOP conditions, and path fragments
/// into required subjects, named entities, and conditions every Claim must
/// satisfy. Plan over the task only. Matches just the server-generated prefixes.
fn planner_question_text(query: &str) -> &str {
    let trimmed = query.trim_start();
    if trimmed.starts_with("Memory Files context:")
        || trimmed.starts_with("Activated procedure context:")
    {
        if let Some((_, task)) = trimmed.rsplit_once("\n\nTask/Question:") {
            let task = task.trim();
            if !task.is_empty() {
                return task;
            }
        }
    }
    query
}

pub(crate) fn plan_answer_pipeline(
    query: &str,
    items: &[RetrievedItem],
    eligible_count: usize,
    candidate_pool_count: usize,
    answer_nodes: Option<usize>,
    detail_level: &str,
    selection_limit: usize,
    full_space_renderers_allowed: bool,
) -> PlannedAnswerPipeline {
    let trace = std::env::var("UMEM_PIPELINE_TRACE").is_ok_and(|value| value == "1");
    let mut stage_started = std::time::Instant::now();
    let mut mark = |stage: &str| {
        if trace {
            eprintln!("PIPELINE_STAGE {stage} {:?}", stage_started.elapsed());
            stage_started = std::time::Instant::now();
        }
    };
    let query_plan = build_query_plan(planner_question_text(query), items);
    mark("query_plan");
    let mut atoms = items
        .iter()
        .flat_map(|item| {
            project_source(ProjectionInput {
                memory_id: item.id,
                content: &item.content,
                source: &item.source,
                entity: item.entity.as_deref(),
                score: item.score,
                is_event: item.event_metadata.is_some()
                    || item.content.trim_start().starts_with("event:"),
            })
        })
        .collect::<Vec<_>>();
    mark("projection");
    rank_evidence_atoms(&query_plan, &mut atoms);
    mark("rank_evidence_atoms");
    let semantic_receipt = apply_semantic_relevance(
        planner_question_text(query),
        &query_plan,
        &mut atoms,
        crate::semantic_scorer::active_scorer().as_deref(),
    );
    mark("semantic_relevance");
    annotate_evidence_chain_candidates(&query_plan, &mut atoms);
    mark("evidence_chains");
    let event_candidates = atoms
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::Event)
        .count();
    let statement_candidates = atoms
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::Statement)
        .count();
    let native_media_candidates = atoms
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::NativeMedia)
        .count();
    let system_metadata_candidates = atoms
        .iter()
        .filter(|atom| {
            matches!(
                atom.kind,
                EvidenceKind::SessionDate | EvidenceKind::SystemMetadata
            )
        })
        .count();
    let original_evidence_candidates = atoms
        .iter()
        .filter(|atom| atom.kind.grounding_role() == GroundingRole::Original)
        .count();
    let context_evidence_candidates = atoms
        .iter()
        .filter(|atom| atom.kind.grounding_role() == GroundingRole::Context)
        .count();
    let expansion_only_candidates = atoms
        .iter()
        .filter(|atom| atom.kind.grounding_role() == GroundingRole::ExpansionOnly)
        .count();
    atoms.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.memory_id.cmp(&right.memory_id))
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
    let candidate_memory_ids = items.iter().map(|item| item.id).collect::<Vec<_>>();
    let effective_selection_limit = grounding_memory_limit(
        &query_plan,
        answer_nodes,
        detail_level,
        selection_limit,
        items.len(),
    );
    let selected_answer_atom_ids =
        select_evidence_atom_ids(&query_plan, &atoms, effective_selection_limit);
    mark("select_evidence");
    // Evidence-chain promotion is deliberately bounded: selected facts must
    // share an explicit subject and a reproducible relation. It never promotes
    // a merely nearby record or a whole document.
    promote_selected_evidence_chains(&query_plan, &mut atoms, &selected_answer_atom_ids);
    mark("promote_chains");
    let rejected_atoms = rejected_evidence_atom_receipts(
        &query_plan,
        &atoms,
        &selected_answer_atom_ids,
        effective_selection_limit,
    );
    mark("rejection_receipts");
    let mut selected_atom_ids =
        expand_selected_evidence_atom_ids(&query_plan, &atoms, &selected_answer_atom_ids);
    if let Some((first, second)) = &query_plan.duration_clauses {
        let scope_memories = atoms
            .iter()
            .filter(|atom| {
                duration_operand_matches(atom, first) || duration_operand_matches(atom, second)
            })
            .map(|atom| atom.memory_id)
            .collect::<HashSet<_>>();
        selected_atom_ids.extend(
            atoms
                .iter()
                .filter(|atom| {
                    duration_operand_matches(atom, first)
                        || duration_operand_matches(atom, second)
                        || (atom.kind == EvidenceKind::SessionDate
                            && scope_memories.contains(&atom.memory_id))
                })
                .map(|atom| atom.atom_id.clone()),
        );
    }
    let rejected_atoms = rejected_atoms
        .into_iter()
        .filter(|atom| !selected_atom_ids.contains(&atom.atom_id))
        .collect();
    let required_scope_atoms = atoms
        .iter()
        .filter(|atom| {
            if let Some((first, second)) = &query_plan.duration_clauses {
                duration_operand_matches(atom, first) || duration_operand_matches(atom, second)
            } else if query_plan.cardinality == Cardinality::Count {
                count_atom_matches_plan(&query_plan, atom)
            } else {
                query_plan.cardinality == Cardinality::List && atom_is_answer_bearing(atom)
            }
        })
        .map(|atom| atom.atom_id.clone())
        .collect::<HashSet<_>>();
    atoms.retain(|atom| selected_atom_ids.contains(&atom.atom_id));
    let runtime_truncated = atoms.len() > MAX_RUNTIME_ATOMS;
    atoms.truncate(MAX_RUNTIME_ATOMS);
    let selected_answer_atom_ids = selected_answer_atom_ids
        .into_iter()
        .filter(|id| atoms.iter().any(|atom| atom.atom_id == *id))
        .collect::<Vec<_>>();
    let selected_memory_ids = selected_memory_ids_for_atoms(&selected_answer_atom_ids, &atoms);
    let selected_set = selected_memory_ids.iter().copied().collect::<HashSet<_>>();
    let dropped_memory_ids = candidate_memory_ids
        .iter()
        .copied()
        .filter(|memory_id| !selected_set.contains(memory_id))
        .collect::<Vec<_>>();
    mark("expand_and_retain");
    let mut slot_coverage =
        assess_slot_coverage(&query_plan, &atoms, eligible_count, candidate_pool_count);
    mark("slot_coverage");
    if runtime_truncated
        || required_scope_atoms
            .iter()
            .any(|id| !atoms.iter().any(|atom| atom.atom_id == *id))
    {
        slot_coverage.push(AnswerSlotCoverageReceipt {
            slot: "proof_scope".into(),
            status: "partial".into(),
            evidence_ids: selected_memory_ids.clone(),
        });
    }
    if query_plan.answer_target == AnswerTarget::Count {
        let counts = count_event_set(&query_plan, &atoms, None);
        if !counts.is_proven() {
            slot_coverage.push(AnswerSlotCoverageReceipt {
                slot: counts
                    .derivation
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "count_proof_unresolved".into()),
                status: "missing".into(),
                evidence_ids: Vec::new(),
            });
        }
    }
    if let Some(proof) = duration_proof(&query_plan, &atoms) {
        let ids = selected_memory_ids_for_atoms(&proof.all_atom_ids(), &atoms);
        for slot in &mut slot_coverage {
            if slot.slot != "proof_scope" {
                slot.status = "covered".into();
                slot.evidence_ids = ids.clone();
            }
        }
    }
    mark("count_and_duration");
    let unified = build_unified_answer(
        query,
        &query_plan,
        &atoms,
        &slot_coverage,
        answer_nodes,
        detail_level,
    );
    mark("unified_answer");
    let document_candidates = items
        .iter()
        .map(|item| item.source.trim())
        .filter(|source| !source.is_empty())
        .collect::<BTreeSet<_>>()
        .len();
    let selected_atoms = atoms
        .iter()
        .filter(|atom| !atom.supports_slots.is_empty())
        .take(MAX_RECEIPT_ATOMS)
        .map(evidence_atom_receipt)
        .collect::<Vec<_>>();
    let selected_bridge_atom_ids = selected_answer_atom_ids
        .iter()
        .filter(|atom_id| {
            atoms
                .iter()
                .find(|atom| &atom.atom_id == *atom_id)
                .is_some_and(|atom| {
                    atom.supports_slots
                        .iter()
                        .any(|slot| slot == "evidence_chain_support")
                })
        })
        .cloned()
        .collect::<Vec<_>>();
    let failure_class = initial_failure_class(
        eligible_count,
        items.len(),
        &slot_coverage,
        unified.is_some(),
    );
    // Every active renderer must provide structured Claim plans. An empty
    // renderer result is a fail-closed answer, never a text-derived fallback.
    let claim_plan = unified
        .as_ref()
        .map(|draft| draft.rendered_claims.clone())
        .unwrap_or_default();
    let operators = build_operator_plan(
        &query_plan,
        full_space_renderers_allowed && requires_full_space_answer_records(query),
    );
    let final_evidence_ids = claim_plan
        .iter()
        .flat_map(|claim| claim.evidence_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let slots_complete = slot_coverage.iter().all(|slot| slot.status == "covered");
    let stage_trace = vec![
        AnswerPipelineStageReceipt {
            stage: "question_plan".to_string(),
            status: "completed".to_string(),
            input_count: 1,
            output_count: query_plan.required_slots.len(),
            evidence_ids: Vec::new(),
            reason: None,
        },
        AnswerPipelineStageReceipt {
            stage: "candidate_freeze".to_string(),
            status: if candidate_pool_count == 0 { "failed" } else { "completed" }.to_string(),
            input_count: candidate_memory_ids.len(),
            output_count: candidate_memory_ids.len(),
            evidence_ids: candidate_memory_ids.clone(),
            reason: (candidate_pool_count == 0)
                .then(|| "no authorized candidates entered the frozen pool".to_string()),
        },
        AnswerPipelineStageReceipt {
            stage: "evidence_plan".to_string(),
            status: if selected_answer_atom_ids.is_empty() {
                "failed"
            } else {
                "completed"
            }
            .to_string(),
            input_count: candidate_memory_ids.len(),
            output_count: selected_answer_atom_ids.len(),
            evidence_ids: selected_memory_ids.clone(),
            reason: selected_answer_atom_ids
                .is_empty()
                .then(|| "no answer-bearing evidence survived slot reranking".to_string()),
        },
        AnswerPipelineStageReceipt {
            stage: "claim_plan".to_string(),
            status: if claim_plan.is_empty() {
                "failed"
            } else if slots_complete {
                "completed"
            } else {
                "partial"
            }
            .to_string(),
            input_count: selected_memory_ids.len(),
            output_count: claim_plan.len(),
            evidence_ids: final_evidence_ids.clone(),
            reason: (!slots_complete)
                .then(|| "one or more required question slots remain incomplete".to_string()),
        },
        AnswerPipelineStageReceipt {
            stage: "independent_verification".to_string(),
            status: "unjudged".to_string(),
            input_count: claim_plan.len(),
            output_count: 0,
            evidence_ids: final_evidence_ids.clone(),
            reason: Some("resolved by the shared local/provider verifier".to_string()),
        },
        AnswerPipelineStageReceipt {
            stage: "citation_binding".to_string(),
            status: "unjudged".to_string(),
            input_count: final_evidence_ids.len(),
            output_count: 0,
            evidence_ids: final_evidence_ids.clone(),
            reason: Some("resolved by the HTTP orchestration layer".to_string()),
        },
        AnswerPipelineStageReceipt {
            stage: "answer".to_string(),
            status: "unjudged".to_string(),
            input_count: claim_plan.len(),
            output_count: usize::from(unified.is_some()),
            evidence_ids: final_evidence_ids.clone(),
            reason: Some("semantic correctness requires answer evaluation".to_string()),
        },
        AnswerPipelineStageReceipt {
            stage: "verified_delivery".to_string(),
            status: "unjudged".to_string(),
            input_count: claim_plan.len(),
            output_count: 0,
            evidence_ids: final_evidence_ids.clone(),
            reason: Some("delivery is rebuilt only from verified Claim coverage".to_string()),
        },
    ];
    let receipt = AnswerPipelineReceipt {
        receipt_format: "answer_pipeline_receipt_v1".to_string(),
        pipeline_version: PIPELINE_VERSION.to_string(),
        query_plan: query_plan_receipt(&query_plan),
        evidence_plan: AnswerEvidencePlanReceipt {
            plan_format: "atom_level_evidence_chain_plan_v7".to_string(),
            source_anchor_format: "answer_evidence_source_anchor_v1".to_string(),
            candidate_pool_count,
            reranked_candidate_count: items.len(),
            rerank_strategy: "typed_slot_conjunction_with_bounded_evidence_chains_v6".to_string(),
            grounding_policy: "selected_original_atoms_with_explicit_relation_bound_proofs"
                .to_string(),
            document_candidates,
            event_candidates,
            statement_candidates,
            native_media_candidates,
            system_metadata_candidates,
            original_evidence_candidates,
            context_evidence_candidates,
            expansion_only_candidates,
            selected_answer_atom_ids,
            selected_bridge_atom_ids,
            selected_atom_count: atoms.len(),
            selected_atoms,
            semantic_scorer: semantic_receipt.as_ref().map(|(backend, _)| backend.clone()),
            semantic_scored_turns: semantic_receipt.as_ref().map(|(_, turns)| *turns),
            candidate_memory_ids,
            selected_memory_ids,
            dropped_memory_ids,
            rejected_atoms,
            selection_limit: effective_selection_limit,
        },
        slot_coverage,
        claim_plan,
        operator_plan: AnswerOperatorPlanReceipt {
            plan_format: "explicit_answer_operator_plan_v1".to_string(),
            planned_operators: operators
                .iter()
                .map(|operator| operator.as_str().to_string())
                .collect(),
            attempted_operators: Vec::new(),
            selected_operator: None,
        },
        stage_trace,
        final_evidence_ids,
        answer_generator: "not_selected".to_string(),
        legacy_fallback_used: false,
        failure_class,
    };

    PlannedAnswerPipeline {
        receipt,
        unified_draft: unified,
        query_plan,
        operators,
        evidence_atoms: atoms,
        source_fingerprints: items
            .iter()
            .map(|item| (item.id, reasoning_source_fingerprint(item)))
            .collect(),
        rejected_source_ids: HashSet::new(),
        general_question_operator: false,
    }
}

fn rejected_evidence_atom_receipts(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    selected_atom_ids: &[String],
    proof_budget: usize,
) -> Vec<AnswerEvidenceRejectionReceipt> {
    let selected = selected_atom_ids
        .iter()
        .map(String::as_str)
        .collect::<HashSet<_>>();
    atoms
        .iter()
        .filter(|atom| atom.kind.grounding_role() == GroundingRole::Original)
        .filter(|atom| !selected.contains(atom.atom_id.as_str()))
        .take(MAX_REJECTION_RECEIPTS)
        .map(|atom| {
            let reason = if plan
                .subject
                .as_deref()
                .is_some_and(|subject| !atom_has_fact_subject(atom, subject))
            {
                "fact_subject_mismatch"
            } else if !atom_can_contribute_to_evidence_chain(plan, atom)
                && !atom_is_answer_bearing(atom)
                && !atom_is_inference_entity_context(atom)
            {
                "no_typed_question_slot"
            } else if selected_atom_ids.len() >= proof_budget {
                "proof_budget_or_coverage_satisfied"
            } else {
                "unlinked_partial_evidence"
            };
            AnswerEvidenceRejectionReceipt {
                atom_id: atom.atom_id.clone(),
                memory_id: atom.memory_id,
                source_anchor: atom.source_anchor.clone(),
                reason: reason.to_string(),
            }
        })
        .collect()
}

fn grounding_memory_limit(
    plan: &QueryPlan,
    _answer_nodes: Option<usize>,
    detail_level: &str,
    requested_limit: usize,
    candidate_count: usize,
) -> usize {
    if candidate_count == 0 {
        return 0;
    }
    let safe_limit = requested_limit.clamp(1, 128).min(candidate_count);
    let automatic = match plan.answer_target {
        AnswerTarget::Count | AnswerTarget::List => match detail_level {
            "concise" => 6,
            "detailed" => 24,
            "custom" => 16,
            _ => 12,
        },
        AnswerTarget::Temporal => 3,
        AnswerTarget::Duration => 2,
        _ if plan.inference_type != InferenceType::Direct => 8,
        _ => 4,
    };
    // `answer_nodes` controls what is displayed, not how many source facts
    // may be required to prove one concise claim. Evidence remains bounded by
    // the explicit source limit and this target-aware proof budget.
    let requested_nodes = automatic.clamp(1, 64);
    let minimum_for_people = plan.people.len().max(1);
    requested_nodes
        .max(minimum_for_people)
        .min(safe_limit)
        .max(1)
}

#[allow(dead_code)]
fn select_evidence_memory_ids(plan: &QueryPlan, atoms: &[EvidenceAtom], limit: usize) -> Vec<u64> {
    if limit == 0 {
        return Vec::new();
    }
    let mut candidates = HashMap::<u64, (f32, HashSet<String>)>::new();
    for atom in atoms {
        let answer_bearing = atom.kind.grounding_role() == GroundingRole::Original
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact");
        if answer_bearing {
            let entry = candidates
                .entry(atom.memory_id)
                .or_insert_with(|| (atom.score, HashSet::new()));
            entry.0 = entry.0.max(atom.score);
            entry.1.extend(atom.supports_slots.iter().cloned());
        }
    }
    if candidates.is_empty() {
        return Vec::new();
    }

    // Context atoms can satisfy a target constraint (for example, a session
    // date for an event) but must never replace the original answer-bearing
    // statement. Keep their slots attached to the owning record.
    for atom in atoms {
        if let Some((_, slots)) = candidates.get_mut(&atom.memory_id) {
            slots.extend(atom.supports_slots.iter().cloned());
        }
    }

    let mut answer_candidates = candidates
        .into_iter()
        .map(|(memory_id, (score, slots))| (memory_id, score, slots))
        .collect::<Vec<_>>();
    answer_candidates.sort_by(|(left_id, left_score, _), (right_id, right_score, _)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_id.cmp(right_id))
    });
    let answer_limit = if !plan.qualifier_terms.is_empty()
        && plan.cardinality == Cardinality::Single
        && plan.answer_shape != AnswerShape::AspectSet
    {
        1
    } else {
        limit
    };
    let required_slots = plan.required_slots.iter().cloned().collect::<HashSet<_>>();
    let primary_score = answer_candidates
        .first()
        .map(|(_, score, _)| *score)
        .unwrap_or_default();
    let score_floor = match plan.answer_shape {
        AnswerShape::AspectSet
        | AnswerShape::ActionSet
        | AnswerShape::Likely
        | AnswerShape::Counterfactual => primary_score - 3.25,
        AnswerShape::CountAggregate => primary_score - 1.75,
        AnswerShape::List | AnswerShape::BoundedList if plan.grounded_list_kind.is_some() => {
            primary_score - 3.25
        }
        AnswerShape::List | AnswerShape::BoundedList if plan.achievement_extremum => {
            primary_score - 3.25
        }
        AnswerShape::List | AnswerShape::BoundedList => primary_score - 1.25,
        _ => primary_score - 1.0,
    };
    let breadth_required = plan.cardinality != Cardinality::Single
        || plan.inference_type != InferenceType::Direct
        || matches!(
            plan.answer_shape,
            AnswerShape::AspectSet | AnswerShape::ActionSet
        );
    let mut selected = Vec::new();
    let mut covered = HashSet::<String>::new();
    while selected.len() < answer_limit {
        let next = answer_candidates
            .iter()
            .filter(|(memory_id, _, _)| !selected.contains(memory_id))
            .map(|(memory_id, score, slots)| {
                let new_slot_count = slots
                    .iter()
                    .filter(|slot| required_slots.contains(*slot) && !covered.contains(*slot))
                    .count();
                let priority = *score + new_slot_count as f32 * 1.35;
                (*memory_id, *score, new_slot_count, priority, slots)
            })
            .max_by(|left, right| {
                left.3
                    .partial_cmp(&right.3)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| left.2.cmp(&right.2))
                    .then_with(|| right.0.cmp(&left.0))
            });
        let Some((memory_id, score, new_slot_count, _, slots)) = next else {
            break;
        };
        if !selected.is_empty() && new_slot_count == 0 {
            if !breadth_required || score < score_floor {
                break;
            }
        }
        selected.push(memory_id);
        covered.extend(slots.iter().cloned());
        if !breadth_required && required_slots.is_subset(&covered) {
            break;
        }
    }
    if plan.qualifier_terms.is_empty() || selected.len() >= limit {
        return selected;
    }

    let required = plan
        .qualifier_terms
        .iter()
        .map(|term| qualifier_slot_name(term))
        .collect::<HashSet<_>>();
    let mut qualifier_covered = atoms
        .iter()
        .filter(|atom| selected.contains(&atom.memory_id))
        .flat_map(|atom| atom.supports_slots.iter())
        .filter(|slot| slot.starts_with("qualifier:"))
        .cloned()
        .collect::<HashSet<_>>();
    if required.is_subset(&qualifier_covered) {
        return selected;
    }

    let mut context_by_memory = HashMap::<u64, (f32, HashSet<String>)>::new();
    for atom in atoms {
        if selected.contains(&atom.memory_id) {
            continue;
        }
        let slots = atom
            .supports_slots
            .iter()
            .filter(|slot| slot.starts_with("qualifier:"))
            .cloned()
            .collect::<HashSet<_>>();
        if slots.is_empty() {
            continue;
        }
        let entry = context_by_memory
            .entry(atom.memory_id)
            .or_insert((atom.score, HashSet::new()));
        entry.0 = entry.0.max(atom.score);
        entry.1.extend(slots);
    }
    let mut contexts = context_by_memory
        .into_iter()
        .map(|(memory_id, (score, slots))| (memory_id, score, slots))
        .collect::<Vec<_>>();
    contexts.sort_by(
        |(left_id, left_score, left_slots), (right_id, right_score, right_slots)| {
            right_slots
                .len()
                .cmp(&left_slots.len())
                .then_with(|| {
                    right_score
                        .partial_cmp(left_score)
                        .unwrap_or(Ordering::Equal)
                })
                .then_with(|| left_id.cmp(right_id))
        },
    );
    for (memory_id, _, slots) in contexts {
        if selected.len() >= limit {
            break;
        }
        let adds_missing = slots
            .iter()
            .any(|slot| required.contains(slot) && !qualifier_covered.contains(slot));
        if !adds_missing {
            continue;
        }
        qualifier_covered.extend(slots);
        selected.push(memory_id);
        if required.is_subset(&qualifier_covered) {
            break;
        }
    }
    selected
}

fn atom_is_answer_bearing(atom: &EvidenceAtom) -> bool {
    atom.kind.grounding_role() == GroundingRole::Original
        && atom
            .supports_slots
            .iter()
            .any(|slot| slot == "answer_bearing_fact")
}

fn atom_is_inference_entity_context(atom: &EvidenceAtom) -> bool {
    atom.kind.grounding_role() == GroundingRole::Original
        && atom
            .supports_slots
            .iter()
            .any(|slot| slot == "inference_entity_context")
}

fn is_multi_entity_inference(plan: &QueryPlan) -> bool {
    plan.inference_type == InferenceType::Likely && plan.named_entities.len() > 1
}

fn grounded_list_request_predicate_term(value: &str) -> bool {
    matches!(
        value,
        "partake"
            | "partakes"
            | "participate"
            | "participates"
            | "participating"
            | "done"
            | "doing"
            | "do"
            | "does"
            | "did"
            | "perform"
            | "performs"
            | "performed"
            | "engage"
            | "engages"
            | "engaged"
    )
}

fn household_reference(value: &str) -> bool {
    word_regex().find_iter(value).any(|term| {
        matches!(
            term.as_str(),
            "family"
                | "families"
                | "fam"
                | "kid"
                | "kids"
                | "child"
                | "children"
                | "husband"
                | "wife"
                | "daughter"
                | "son"
                | "parent"
                | "parents"
                | "sibling"
                | "siblings"
                | "spouse"
        )
    }) || contains_any(value, &["家人", "家庭", "孩子", "子女", "配偶"])
}

fn query_requests_household_list_context(query_lower: &str) -> bool {
    household_reference(query_lower)
}

fn query_requests_household_dependent_preference(query_lower: &str) -> bool {
    static POSSESSIVE_DEPENDENT: OnceLock<regex::Regex> = OnceLock::new();
    let possessive_dependent = POSSESSIVE_DEPENDENT.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b[a-z][a-z'’-]{0,63}(?:'s|’s)\s+(?:(?:\d+|one|two|three|four)\s+)?(?:(?:older|younger)\s+)?(?:kids?|child|children|daughter|son)\b",
        )
        .expect("household dependent query regex")
    });
    if !possessive_dependent.is_match(query_lower) {
        return false;
    }
    let terms = word_regex()
        .find_iter(query_lower)
        .map(|term| term.as_str().to_ascii_lowercase())
        .collect::<Vec<_>>();
    terms.iter().enumerate().any(|(index, term)| {
        matches!(
            term.as_str(),
            "like"
                | "likes"
                | "liked"
                | "love"
                | "loves"
                | "loved"
                | "enjoy"
                | "enjoys"
                | "enjoyed"
                | "prefer"
                | "prefers"
                | "preferred"
        ) && !(term == "like" && index > 0 && terms[index - 1] == "look")
    })
}

fn grounded_list_kind_implies_list_cardinality(
    kind: crate::memory_engine_summary::GroundedListKind,
    query_lower: &str,
) -> bool {
    use crate::memory_engine_summary::GroundedListKind;

    match kind {
        // Event and activity extractors are also useful for a singular
        // question about one event. Only promote them when the wording itself
        // asks for multiple values; otherwise a shared-event answer gets
        // flattened into an unrelated list.
        GroundedListKind::Events => contains_any(
            query_lower,
            &[
                "what events",
                "which events",
                "events has",
                "events have",
                "events did",
                "in what ways",
                "what ways",
                "哪些事件",
                "参加过什么活动",
                "如何参与社区",
            ],
        ),
        GroundedListKind::Activities => contains_any(
            query_lower,
            &[
                "what activities",
                "which activities",
                "creative projects",
                "art projects",
                "哪些活动",
                "参与哪些",
            ],
        ),
        GroundedListKind::Fields
        | GroundedListKind::Books
        | GroundedListKind::Locations
        | GroundedListKind::Preferences
        | GroundedListKind::Destress => true,
    }
}

fn direct_household_activity_relation(statement_lower: &str) -> bool {
    static DIRECT_HOUSEHOLD_ACTIVITY: OnceLock<regex::Regex> = OnceLock::new();
    let direct_pattern = DIRECT_HOUSEHOLD_ACTIVITY.get_or_init(|| {
        regex::Regex::new(
            r"\b(?:(?:with|along\s+with)\s+(?:(?:my|our|the)\s+)?(?:family|fam|kids?|children|husband|wife|daughter|son|parents?|siblings?|spouse)|(?:my|our)\s+(?:family|fam|kids?|children)\s+and\s+i|(?:i|we)\s+(?:\w+\s+){0,5}(?:took|take|bringing|bring|brought|spent|spend|went|go|visited|visit)\s+(?:(?:my|our|the)\s+)?(?:family|fam|kids?|children)|(?:my|our)\s+(?:family|fam)\s+(?:\w+\s+){0,2}(?:trip|outing|camp(?:ing)?|hike|visit))\b",
        )
        .expect("direct household activity regex")
    });
    direct_pattern.is_match(statement_lower)
        || contains_any(
            statement_lower,
            &["和家人", "与家人", "带孩子", "带着孩子", "全家", "一家人"],
        )
}

fn household_question_binds_collective_response(question: &EvidenceAtom) -> bool {
    let question_lower = question.text.to_lowercase();
    household_reference(&question_lower)
        && contains_any(
            &question_lower,
            &[
                "with them",
                "with the kids",
                "with your kids",
                "with your family",
                "together",
                "和家人",
                "和孩子",
                "一起",
            ],
        )
}

fn list_value_has_household_context(
    statement_lower: &str,
    adjacent_question: Option<&EvidenceAtom>,
) -> bool {
    if direct_household_activity_relation(statement_lower) {
        return true;
    }
    let collective_response = contains_any(statement_lower, &["we ", "we\u{27}", "our ", "我们"])
        && contains_any(
            statement_lower,
            &["together", "both", "each other", "一起", "共同"],
        );
    collective_response
        && adjacent_question.is_some_and(household_question_binds_collective_response)
}

fn atom_grounded_list_values(plan: &QueryPlan, atom: &EvidenceAtom) -> Vec<String> {
    let mut values = plan
        .grounded_list_kind
        .map(|kind| {
            crate::memory_engine_summary::answer_list_values_from_statement(kind, &atom.text)
        })
        .unwrap_or_default();
    if plan.grounded_list_kind == Some(crate::memory_engine_summary::GroundedListKind::Destress)
        && crate::memory_engine_summary::decompression_evidence_strength(&atom.text)
            < destress_query_minimum_relation_strength(plan)
    {
        values.clear();
    }
    values
}

fn destress_query_minimum_relation_strength(plan: &QueryPlan) -> u8 {
    let asks_for_explicit_stress_relief = plan
        .object_terms
        .iter()
        .chain(plan.query_terms.iter())
        .any(|term| {
            matches!(
                term.as_str(),
                "destress" | "de-stress" | "stress" | "stressrelief" | "减压"
            )
        });
    if asks_for_explicit_stress_relief {
        3
    } else {
        2
    }
}

fn same_turn_destress_activity_values(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    turn_context_lower: &str,
) -> Vec<String> {
    if plan.grounded_list_kind != Some(crate::memory_engine_summary::GroundedListKind::Destress)
        || atom.kind.grounding_role() != GroundingRole::Original
        || !statement_records_observed_action(&atom.text)
        || crate::memory_engine_summary::decompression_evidence_strength(turn_context_lower)
            < destress_query_minimum_relation_strength(plan)
    {
        return Vec::new();
    }

    crate::memory_engine_summary::answer_list_values_from_statement(
        crate::memory_engine_summary::GroundedListKind::Activities,
        &atom.text,
    )
}

fn destress_relation_strength(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    turn_context_lower: &str,
) -> u8 {
    if plan.grounded_list_kind != Some(crate::memory_engine_summary::GroundedListKind::Destress) {
        return 0;
    }
    let mut strength =
        crate::memory_engine_summary::decompression_evidence_strength(turn_context_lower);
    // "for me" binds therapeutic language to the requested person's own
    // activity, rather than a collective or adjacent wellbeing statement.
    if strength >= 3
        && atom_speaker_is_plan_subject(plan, atom)
        && turn_context_lower.contains("for me")
    {
        strength = strength.saturating_add(1);
    }
    strength
}

fn atom_grounded_list_values_with_same_turn_context(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    turn_context_lower: &str,
) -> (Vec<String>, bool) {
    let direct_values = atom_grounded_list_values(plan, atom);
    if !direct_values.is_empty() {
        return (direct_values, false);
    }
    let bridged_values = same_turn_destress_activity_values(plan, atom, turn_context_lower);
    let bridge_used = !bridged_values.is_empty();
    (bridged_values, bridge_used)
}

fn direct_household_dependent_reference(statement_lower: &str) -> bool {
    static DEPENDENT_REFERENCE: OnceLock<regex::Regex> = OnceLock::new();
    let dependent_reference = DEPENDENT_REFERENCE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b((?:(?:my|our|the|these|those|their|her|his|your)\s+)?(?:(?:\d+|one|two|three|four)\s+)?(?:(?:older|younger)\s+)?)(?:kids?|child|children|daughter|son)\b",
        )
        .expect("household dependent reference regex")
    });
    dependent_reference
        .captures_iter(statement_lower)
        .any(|captures| {
            captures
                .get(1)
                .is_some_and(|prefix| !prefix.as_str().trim().is_empty())
        })
}

fn direct_household_dependent_preference_subject(statement_lower: &str) -> bool {
    static DEPENDENT_PREFERENCE: OnceLock<regex::Regex> = OnceLock::new();
    let dependent_preference = DEPENDENT_PREFERENCE.get_or_init(|| {
        regex::Regex::new(
            r"(?i)\b((?:(?:my|our|the|these|those|their|her|his|your)\s+)?(?:(?:\d+|one|two|three|four)\s+)?(?:(?:older|younger)\s+)?)(?:kids?|child|children|daughter|son)\s+(?:(?:(?:really|absolutely|especially|always)\s+)*(?:love|loves|loved|like|likes|liked|enjoy|enjoys|enjoyed|prefer|prefers|preferred)|(?:am|are|is|was|were)\s+(?:(?:really|especially)\s+)*(?:stoked\s+for|excited\s+about))\b",
        )
        .expect("household dependent preference regex")
    });
    dependent_preference
        .captures_iter(statement_lower)
        .any(|captures| {
            captures
                .get(1)
                .is_some_and(|prefix| !prefix.as_str().trim().is_empty())
        })
}

fn dependent_preference_uses_pronoun(statement_lower: &str) -> bool {
    static DEPENDENT_PRONOUN_PREFERENCE: OnceLock<regex::Regex> = OnceLock::new();
    DEPENDENT_PRONOUN_PREFERENCE
        .get_or_init(|| {
            regex::Regex::new(
                r"(?i)^\s*they\s+(?:(?:(?:really|absolutely|especially|always)\s+)*(?:love|loves|loved|like|likes|liked|enjoy|enjoys|enjoyed|prefer|prefers|preferred)|(?:am|are|were)\s+(?:(?:really|especially)\s+)*(?:stoked\s+for|excited\s+about))\b",
            )
            .expect("dependent preference pronoun regex")
        })
        .is_match(statement_lower)
}

fn atom_speaker_is_plan_subject(plan: &QueryPlan, atom: &EvidenceAtom) -> bool {
    plan.subject.as_deref().is_some_and(|subject| {
        atom.speaker
            .as_deref()
            .is_some_and(|speaker| normalize_person_key(speaker) == normalize_person_key(subject))
    })
}

fn statement_explicitly_names_plan_owner(plan: &QueryPlan, statement_lower: &str) -> bool {
    plan.subject.as_deref().is_some_and(|subject| {
        let owner = subject.to_lowercase();
        statement_lower.contains(&format!("{owner}'s"))
            || statement_lower.contains(&format!("{owner}’s"))
    })
}

fn has_bounded_household_dependent_antecedent(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    snapshot: &[EvidenceAtom],
) -> bool {
    if atom.context_group == usize::MAX || !atom_speaker_is_plan_subject(plan, atom) {
        return false;
    }
    snapshot.iter().any(|candidate| {
        candidate.memory_id == atom.memory_id
            && candidate.kind.grounding_role() == GroundingRole::Original
            && candidate.context_group != usize::MAX
            && candidate.context_group < atom.context_group
            // Resolve a dependent pronoun only from the immediately local
            // dialogue context. A broad source-level match leaks unrelated
            // preferences from the owner into the dependent's answer.
            && atom.context_group - candidate.context_group <= 2
            && atom_speaker_is_plan_subject(plan, candidate)
            && direct_household_dependent_reference(&candidate.text.to_lowercase())
    })
}

fn household_dependent_preference_value(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    snapshot: &[EvidenceAtom],
) -> bool {
    if !plan.household_dependent_preference
        || plan.grounded_list_kind
            != Some(crate::memory_engine_summary::GroundedListKind::Preferences)
        || crate::memory_engine_summary::answer_list_values_from_statement(
            crate::memory_engine_summary::GroundedListKind::Preferences,
            &atom.text,
        )
        .is_empty()
    {
        return false;
    }

    let statement_lower = atom.text.to_lowercase();
    (direct_household_dependent_preference_subject(&statement_lower)
        && (atom_speaker_is_plan_subject(plan, atom)
            || statement_explicitly_names_plan_owner(plan, &statement_lower)))
        || (dependent_preference_uses_pronoun(&statement_lower)
            && has_bounded_household_dependent_antecedent(plan, atom, snapshot))
}

fn typed_list_value_keys(plan: &QueryPlan, atom: &EvidenceAtom) -> BTreeSet<String> {
    let mut values = atom_grounded_list_values(plan, atom)
        .into_iter()
        .map(|value| normalize_for_dedup(&value))
        .filter(|value| !value.is_empty())
        .collect::<BTreeSet<_>>();
    values.extend(atom.supports_slots.iter().filter_map(|slot| {
        slot.strip_prefix("list_value_key:")
            .map(normalize_for_dedup)
            .filter(|value| !value.is_empty())
    }));
    values
}

/// Open typed lists may need complementary values from a lower-ranked record.
/// Once an atom independently passes the subject, answer-bearing-fact, and
/// typed-value checks, score distance alone must not erase that distinct value.
/// The exception is deliberately limited to open lists; bounded lists retain
/// their stricter score and quantity rules.
fn admissible_open_typed_list_value_candidate(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    score_floor: f32,
) -> bool {
    let independently_validated = plan.grounded_list_kind.is_some()
        && atom.kind.grounding_role() == GroundingRole::Original
        && atom_is_answer_bearing(atom)
        && atom.supports_slots.iter().any(|slot| slot == "list_value")
        && !typed_list_value_keys(plan, atom).is_empty();
    independently_validated
        && (atom.score >= score_floor
            || (plan.cardinality == Cardinality::List && plan.list_scope == "open"))
}

fn has_admissible_new_typed_list_value(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    selected_list_values: &HashSet<String>,
    score_floor: f32,
) -> bool {
    admissible_open_typed_list_value_candidate(plan, atom, score_floor)
        && typed_list_value_keys(plan, atom)
            .iter()
            .any(|value| !selected_list_values.contains(value))
}

fn typed_list_relation_strength(plan: &QueryPlan, atom: &EvidenceAtom) -> u8 {
    if plan.grounded_list_kind != Some(crate::memory_engine_summary::GroundedListKind::Destress) {
        return 0;
    }
    atom.supports_slots
        .iter()
        .filter_map(|slot| slot.strip_prefix("decompression_strength:"))
        .filter_map(|value| value.parse::<u8>().ok())
        .max()
        .unwrap_or_default()
}

fn list_value_participant_scope_matches(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    statement_lower: &str,
    snapshot: &[EvidenceAtom],
    adjacent_question: Option<&EvidenceAtom>,
) -> bool {
    if plan.household_dependent_preference {
        return household_dependent_preference_value(plan, atom, snapshot);
    }
    !plan.list_requires_household_context
        || list_value_has_household_context(statement_lower, adjacent_question)
}

fn requested_bounded_list_value_count(plan: &QueryPlan) -> Option<usize> {
    (plan.cardinality == Cardinality::List && plan.list_scope == "bounded").then(|| {
        plan.quantity_terms
            .iter()
            .filter_map(|value| explicit_quantity_value(value))
            .max()
            .unwrap_or(1)
    })
}

fn coordinated_proper_value_count(value: &str) -> usize {
    usize::from(has_coordinated_proper_value_pair(value)) * 2
}

fn atom_covers_complete_bounded_list(plan: &QueryPlan, atom: &EvidenceAtom) -> bool {
    requested_bounded_list_value_count(plan).is_some_and(|requested| {
        atom_is_answer_bearing(atom)
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "bounded_list_response_bridge")
            && coordinated_proper_value_count(&atom.text) >= requested
    })
}

/// Select original facts, rather than whole records. A record can contain a
/// correct sentence beside a more generic one; choosing the record alone loses
/// that distinction before claim planning.
fn select_evidence_atom_ids(plan: &QueryPlan, atoms: &[EvidenceAtom], limit: usize) -> Vec<String> {
    if limit == 0 {
        return Vec::new();
    }
    let mut answer_candidates = atoms
        .iter()
        .filter(|atom| atom_is_answer_bearing(atom) || atom_is_inference_entity_context(atom))
        .collect::<Vec<_>>();
    let mut candidate_ids = answer_candidates
        .iter()
        .map(|atom| atom.atom_id.as_str())
        .collect::<HashSet<_>>();
    // A fact can be indispensable without independently answering the whole
    // question. Admit only bounded, typed partial facts; promotion still
    // requires an explicit relation before they can become answer evidence.
    for atom in atoms.iter().filter(|atom| {
        // Direct multi-hop questions need partial source facts. Inference has
        // its own typed premise path; admitting every partial direct-looking
        // statement there dilutes the established inference-basis selector.
        plan.inference_type == InferenceType::Direct
            && atom_can_contribute_to_evidence_chain(plan, atom)
    }) {
        if candidate_ids.insert(atom.atom_id.as_str()) {
            answer_candidates.push(atom);
        }
    }
    if answer_candidates.is_empty() {
        return Vec::new();
    }
    // Session-summary questions should ground on the authored summary
    // statement when one is available. Ingest also creates an Event wrapper
    // containing the same text; allowing that wrapper's duplicated keywords
    // to win produces a citation such as `source=...` instead of the compact
    // summary fact the user requested. This preference is scoped to explicit
    // summary wording and does not change ordinary event retrieval.
    let summary_request = plan.object_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "summary" | "summaries" | "overview" | "synopsis" | "概要" | "摘要"
        )
    });
    let selection_score = |atom: &EvidenceAtom| {
        if summary_request && atom.kind == EvidenceKind::Event {
            atom.score - 2.0
        } else {
            atom.score
        }
    };
    answer_candidates.sort_by(|left, right| {
        selection_score(right)
            .partial_cmp(&selection_score(left))
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.score.partial_cmp(&left.score).unwrap_or(Ordering::Equal))
            .then_with(|| left.memory_id.cmp(&right.memory_id))
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });

    if let Some(complete) = answer_candidates
        .iter()
        .find(|atom| atom_covers_complete_bounded_list(plan, atom))
    {
        return vec![complete.atom_id.clone()];
    }

    // A scalar output may require several source atoms, but that is not a
    // license to keep nearby facts. Expand the proof budget only when the
    // query itself requires multiple scopes, inference, or a pair that has an
    // explicit relation and jointly closes the required slots. The renderer
    // still applies the user-facing node budget later.
    let answer_limit = if scalar_answer_needs_complementary_proof(plan, &answer_candidates) {
        limit
    } else {
        1.min(limit)
    };
    let required_slots = plan.required_slots.iter().cloned().collect::<HashSet<_>>();
    let primary_score = answer_candidates
        .first()
        .map(|atom| atom.score)
        .unwrap_or_default();
    let score_floor = match plan.answer_shape {
        AnswerShape::AspectSet
        | AnswerShape::ActionSet
        | AnswerShape::Likely
        | AnswerShape::Counterfactual => primary_score - 3.25,
        AnswerShape::CountAggregate => primary_score - 1.75,
        AnswerShape::List | AnswerShape::BoundedList if plan.grounded_list_kind.is_some() => {
            primary_score - 3.25
        }
        AnswerShape::List | AnswerShape::BoundedList if plan.achievement_extremum => {
            primary_score - 3.25
        }
        AnswerShape::List | AnswerShape::BoundedList => primary_score - 1.25,
        _ => primary_score - 1.0,
    };
    let breadth_required = plan.cardinality != Cardinality::Single
        || plan.inference_type != InferenceType::Direct
        || matches!(
            plan.answer_shape,
            AnswerShape::AspectSet | AnswerShape::ActionSet
        );
    let source_diversity_required = plan.inference_type != InferenceType::Direct
        || matches!(
            plan.answer_shape,
            AnswerShape::ActionSet | AnswerShape::CountAggregate
        );
    let mut selected = Vec::<&EvidenceAtom>::new();
    let breadth_required = breadth_required || plan.recommendation_requested;
    let source_diversity_required = source_diversity_required || plan.recommendation_requested;
    let mut selected_ids = HashSet::<String>::new();
    let mut selected_memory_ids = HashSet::<u64>::new();
    let mut selected_contexts = HashSet::<(u64, usize)>::new();
    let mut selected_list_values = HashSet::<String>::new();
    let mut covered = HashSet::<String>::new();

    while selected.len() < answer_limit {
        let new_typed_list_value_candidate_available = plan.grounded_list_kind.is_some()
            && answer_candidates.iter().any(|candidate| {
                !selected_ids.contains(&candidate.atom_id)
                    && has_admissible_new_typed_list_value(
                        plan,
                        candidate,
                        &selected_list_values,
                        score_floor,
                    )
            });
        let distinct_source_candidate_available = source_diversity_required
            && !selected.is_empty()
            && answer_candidates.iter().any(|candidate| {
                !selected_ids.contains(&candidate.atom_id)
                    && candidate.score >= score_floor
                    && !selected_memory_ids.contains(&candidate.memory_id)
            });
        let distinct_context_candidate_available = breadth_required
            && !selected.is_empty()
            && answer_candidates.iter().any(|candidate| {
                !selected_ids.contains(&candidate.atom_id)
                    && candidate.score >= score_floor
                    && (!distinct_source_candidate_available
                        || !selected_memory_ids.contains(&candidate.memory_id))
                    && (candidate.context_group == usize::MAX
                        || !selected_contexts
                            .contains(&(candidate.memory_id, candidate.context_group)))
            });
        let next = answer_candidates
            .iter()
            .copied()
            .filter(|atom| {
                !selected_ids.contains(&atom.atom_id)
                    && (!new_typed_list_value_candidate_available
                        || has_admissible_new_typed_list_value(
                            plan,
                            atom,
                            &selected_list_values,
                            score_floor,
                        ))
                    && (!distinct_source_candidate_available
                        || !selected_memory_ids.contains(&atom.memory_id))
                    && (!distinct_context_candidate_available
                        || atom.context_group == usize::MAX
                        || !selected_contexts.contains(&(atom.memory_id, atom.context_group)))
            })
            .map(|atom| {
                let candidate_slots = evidence_chain_slots(plan, atom);
                let new_slot_count = candidate_slots
                    .iter()
                    .filter(|slot| required_slots.contains(*slot) && !covered.contains(*slot))
                    .count();
                let novel_list_value_count = typed_list_value_keys(plan, atom)
                    .iter()
                    .filter(|value| !selected_list_values.contains(*value))
                    .count();
                let typed_relation_strength = typed_list_relation_strength(plan, atom);
                let priority = selection_score(atom)
                    + new_slot_count as f32 * 1.35
                    + novel_list_value_count as f32 * 0.95;
                (atom, new_slot_count, typed_relation_strength, priority)
            })
            .max_by(|left, right| {
                left.2
                    .cmp(&right.2)
                    .then_with(|| left.3.partial_cmp(&right.3).unwrap_or(Ordering::Equal))
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| right.0.memory_id.cmp(&left.0.memory_id))
                    .then_with(|| right.0.ordinal.cmp(&left.0.ordinal))
            });
        let Some((atom, new_slot_count, _, _)) = next else {
            break;
        };
        let explicit_complementary_extremum = plan.achievement_extremum
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "achievement_extremum");
        let complementary_typed_list_value =
            has_admissible_new_typed_list_value(plan, atom, &selected_list_values, score_floor);
        if !selected.is_empty()
            && new_slot_count == 0
            && !explicit_complementary_extremum
            && !complementary_typed_list_value
        {
            if !breadth_required || atom.score < score_floor {
                break;
            }
        }
        covered.extend(evidence_chain_slots(plan, atom));
        selected_list_values.extend(typed_list_value_keys(plan, atom));
        selected_ids.insert(atom.atom_id.clone());
        selected_memory_ids.insert(atom.memory_id);
        if atom.context_group != usize::MAX {
            selected_contexts.insert((atom.memory_id, atom.context_group));
        }
        selected.push(atom);
        if required_slots.is_subset(&covered)
            && (!breadth_required || plan.structural_selectors.len() > 1)
        {
            break;
        }
    }

    // Qualifier facts may live in a different record. Add only the original
    // atom that covers a missing qualifier, while honoring the document limit.
    if !plan.qualifier_terms.is_empty() {
        let required_qualifiers = plan
            .qualifier_terms
            .iter()
            .map(|term| qualifier_slot_name(term))
            .collect::<HashSet<_>>();
        let mut qualifier_covered = selected
            .iter()
            .flat_map(|atom| atom.supports_slots.iter())
            .filter(|slot| slot.starts_with("qualifier:"))
            .cloned()
            .collect::<HashSet<_>>();
        let mut selected_memories = selected
            .iter()
            .map(|atom| atom.memory_id)
            .collect::<HashSet<_>>();
        let mut qualifier_candidates = atoms
            .iter()
            .filter(|atom| {
                atom.kind.grounding_role() == GroundingRole::Original
                    && !selected_ids.contains(&atom.atom_id)
                    && atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot.starts_with("qualifier:"))
            })
            .collect::<Vec<_>>();
        qualifier_candidates.sort_by(|left, right| {
            let left_missing = left
                .supports_slots
                .iter()
                .filter(|slot| {
                    required_qualifiers.contains(*slot) && !qualifier_covered.contains(*slot)
                })
                .count();
            let right_missing = right
                .supports_slots
                .iter()
                .filter(|slot| {
                    required_qualifiers.contains(*slot) && !qualifier_covered.contains(*slot)
                })
                .count();
            right_missing.cmp(&left_missing).then_with(|| {
                right
                    .score
                    .partial_cmp(&left.score)
                    .unwrap_or(Ordering::Equal)
            })
        });
        for atom in qualifier_candidates {
            if required_qualifiers.is_subset(&qualifier_covered) {
                break;
            }
            let adds_missing = atom.supports_slots.iter().any(|slot| {
                required_qualifiers.contains(slot) && !qualifier_covered.contains(slot)
            });
            if !adds_missing {
                continue;
            }
            if !selected_memories.contains(&atom.memory_id) && selected_memories.len() >= limit {
                continue;
            }
            qualifier_covered.extend(
                atom.supports_slots
                    .iter()
                    .filter(|slot| slot.starts_with("qualifier:"))
                    .cloned(),
            );
            selected_memories.insert(atom.memory_id);
            selected_ids.insert(atom.atom_id.clone());
            selected.push(atom);
        }
    }

    selected
        .into_iter()
        .map(|atom| atom.atom_id.clone())
        .collect()
}

fn scalar_answer_needs_complementary_proof(plan: &QueryPlan, candidates: &[&EvidenceAtom]) -> bool {
    if plan.cardinality != Cardinality::Single
        || plan.inference_type != InferenceType::Direct
        || matches!(
            plan.answer_shape,
            AnswerShape::AspectSet | AnswerShape::ActionSet
        )
        || plan.structural_selectors.len() > 1
    {
        return true;
    }
    candidates.iter().enumerate().any(|(index, left)| {
        candidates.iter().skip(index + 1).any(|right| {
            !evidence_chain_relation(plan, left, right).is_empty()
                && evidence_chain_covers_answer(
                    plan,
                    &evidence_chain_slots(plan, left)
                        .union(&evidence_chain_slots(plan, right))
                        .cloned()
                        .collect(),
                )
        })
    })
}

fn annotate_evidence_chain_candidates(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    for atom in atoms {
        if !atom_can_contribute_to_evidence_chain(plan, atom) {
            continue;
        }
        let mut slots = atom.supports_slots.iter().cloned().collect::<BTreeSet<_>>();
        slots.insert("evidence_chain_candidate".to_string());
        atom.supports_slots = slots.into_iter().collect();
    }
}

fn atom_can_contribute_to_evidence_chain(plan: &QueryPlan, atom: &EvidenceAtom) -> bool {
    if atom.kind.grounding_role() != GroundingRole::Original
        || is_low_information_statement(&atom.text)
    {
        return false;
    }
    if plan
        .subject
        .as_deref()
        .is_some_and(|subject| !atom_has_fact_subject(atom, subject))
    {
        return false;
    }
    let slots = evidence_chain_slots(plan, atom);
    slots.iter().any(|slot| {
        matches!(
            slot.as_str(),
            "action"
                | "object"
                | "location"
                | "event_time"
                | "qualifier_context"
                | "answer_class"
                | "list_value"
                | "inference_basis"
        ) || slot.starts_with("qualifier:")
            || slot.starts_with("named_entity:")
    })
}

fn evidence_chain_slots(plan: &QueryPlan, atom: &EvidenceAtom) -> BTreeSet<String> {
    let mut slots = atom
        .supports_slots
        .iter()
        .filter(|slot| {
            plan.required_slots.contains(*slot)
                || matches!(
                    slot.as_str(),
                    "action" | "object" | "location" | "event_time" | "answer_class" | "list_value"
                )
                || slot.starts_with("person:")
                || slot.starts_with("qualifier:")
                || slot.starts_with("named_entity:")
        })
        .cloned()
        .collect::<BTreeSet<_>>();
    if atom.kind.grounding_role() != GroundingRole::Original {
        return slots;
    }

    let lower = atom.text.to_lowercase();
    let subject_bound = plan
        .subject
        .as_deref()
        .is_none_or(|subject| atom_has_fact_subject(atom, subject));
    if subject_bound && plan.subject.is_some() {
        slots.insert("subject".to_string());
    }
    for person in &plan.people {
        if atom_matches_person(atom, &lower, person, false) {
            slots.insert(person_slot_name(person));
        }
    }
    if subject_bound && statement_has_direct_action_predicate(plan, &lower) {
        slots.insert("action".to_string());
    }
    if subject_bound
        && !plan.object_terms.is_empty()
        && matching_lexical_term_count(&lower, &plan.object_terms)
            >= required_object_overlap(plan.object_terms.len())
    {
        slots.insert("object".to_string());
    }
    if subject_bound
        && plan.answer_target == AnswerTarget::Location
        && atom_has_location_value(plan, &atom.text)
    {
        slots.insert("location".to_string());
    }
    if subject_bound && plan.answer_target == AnswerTarget::Temporal {
        if contains_date_like_value(&lower) || contains_bare_calendar_month_reference(&lower) {
            slots.insert("event_time".to_string());
            slots.insert("temporal_scope".to_string());
        }
    }
    if subject_bound && !plan.qualifier_terms.is_empty() {
        let matched = plan
            .qualifier_terms
            .iter()
            .filter(|term| text_matches_lexical_term(&lower, term))
            .collect::<Vec<_>>();
        if !matched.is_empty() {
            slots.insert("qualifier_context".to_string());
            for term in matched {
                slots.insert(qualifier_slot_name(term));
            }
        }
    }
    slots
}

fn evidence_chain_relation(
    plan: &QueryPlan,
    left: &EvidenceAtom,
    right: &EvidenceAtom,
) -> Vec<String> {
    let shared_subjects = left
        .fact_subjects
        .iter()
        .filter_map(|subject| {
            let key = normalize_person_key(subject);
            right
                .fact_subjects
                .iter()
                .any(|candidate| normalize_person_key(candidate) == key)
                .then_some(key)
        })
        .collect::<BTreeSet<_>>();
    if shared_subjects.is_empty() {
        return Vec::new();
    }

    let left_dates = evidence_chain_explicit_dates(left);
    let right_dates = evidence_chain_explicit_dates(right);
    if !left_dates.is_empty() && !right_dates.is_empty() && left_dates != right_dates {
        return Vec::new();
    }
    // Lexical association can expand candidates, but cannot create a proof
    // edge. Only an independently resolved, exact atom reference can do so.
    let bound_reference = count_event_bridge_anchor(left)
        .is_some_and(|anchor| anchor == right.atom_id)
        || count_event_bridge_anchor(right).is_some_and(|anchor| anchor == left.atom_id);
    let possessive_reference =
        explicit_possessive_reference(left, right) || explicit_possessive_reference(right, left);
    if !bound_reference && !possessive_reference {
        return Vec::new();
    }

    // Preserve shared context for diagnostics, never as the proof itself.
    let shared_phrases = evidence_chain_relation_phrases(left)
        .intersection(&evidence_chain_relation_phrases(right))
        .cloned()
        .collect::<Vec<_>>();
    let left_lower = left.text.to_lowercase();
    let right_lower = right.text.to_lowercase();
    let shared_entities = plan
        .named_entities
        .iter()
        .filter(|entity| {
            text_matches_named_entity(&left_lower, entity)
                && text_matches_named_entity(&right_lower, entity)
        })
        .map(|entity| normalize_for_dedup(entity))
        .collect::<Vec<_>>();
    let shared_dates = evidence_chain_explicit_dates(left)
        .intersection(&evidence_chain_explicit_dates(right))
        .cloned()
        .collect::<Vec<_>>();
    let mut reasons = shared_subjects
        .iter()
        .map(|subject| format!("same_fact_subject:{subject}"))
        .collect::<Vec<_>>();
    let reference_kind = if bound_reference {
        "bound_atom_reference"
    } else {
        "possessive_apposition"
    };
    reasons.push(format!(
        "{reference_kind}:{}:{}",
        left.atom_id, right.atom_id
    ));
    reasons.extend(
        shared_phrases
            .into_iter()
            .map(|phrase| format!("shared_relation_phrase:{phrase}")),
    );
    reasons.extend(
        shared_entities
            .into_iter()
            .map(|entity| format!("shared_named_entity:{entity}")),
    );
    reasons.extend(
        shared_dates
            .into_iter()
            .map(|date| format!("shared_event_date:{date}")),
    );
    reasons
}

fn explicit_possessive_reference(reference: &EvidenceAtom, definition: &EvidenceAtom) -> bool {
    if !reference
        .speaker
        .as_deref()
        .zip(definition.speaker.as_deref())
        .is_some_and(|(left, right)| normalize_person_key(left) == normalize_person_key(right))
    {
        return false;
    }
    static APPOSITION: OnceLock<Regex> = OnceLock::new();
    let regex = APPOSITION.get_or_init(|| {
        Regex::new(
        r"\b(?i:my)\s+([a-z]+(?:\s+[a-z]+){0,3}),\s*([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*),"
    ).expect("possessive apposition regex")
    });
    regex.captures_iter(&definition.text).any(|capture| {
        let property = format!("my {}", &capture[1]);
        contains_bounded_ascii_phrase(&reference.text.to_lowercase(), &property)
    })
}

fn evidence_chain_relation_phrases(atom: &EvidenceAtom) -> BTreeSet<String> {
    let fact_subjects = atom
        .fact_subjects
        .iter()
        .map(|subject| normalize_for_dedup(subject))
        .collect::<HashSet<_>>();
    let terms = word_regex()
        .find_iter(&atom.text)
        .map(|value| normalize_for_dedup(value.as_str()))
        .collect::<Vec<_>>();
    terms
        .windows(2)
        .filter(|pair| {
            pair.iter().all(|term| {
                term.chars().count() >= 3
                    && !fact_subjects.contains(term)
                    && evidence_chain_relation_token_is_specific(term)
            })
        })
        .map(|pair| format!("{}+{}", pair[0], pair[1]))
        .collect()
}

fn evidence_chain_relation_token_is_specific(term: &str) -> bool {
    !matches!(
        term,
        "about"
            | "after"
            | "and"
            | "because"
            | "been"
            | "from"
            | "have"
            | "just"
            | "like"
            | "that"
            | "the"
            | "their"
            | "this"
            | "they"
            | "was"
            | "were"
            | "with"
            | "would"
            | "your"
    )
}

fn evidence_chain_explicit_dates(atom: &EvidenceAtom) -> BTreeSet<String> {
    parse_temporal_dates(&atom.text)
        .into_iter()
        .map(|date| date.to_string())
        .collect()
}

fn evidence_chain_covers_answer(plan: &QueryPlan, slots: &BTreeSet<String>) -> bool {
    if plan.subject.is_some() && !slots.contains("subject") {
        return false;
    }
    if plan.answer_target == AnswerTarget::Location {
        return slots.contains("location")
            && (plan.action_terms.is_empty() || slots.contains("action"));
    }
    if plan.answer_target == AnswerTarget::Temporal {
        return slots.contains("event_time")
            && (plan.action_terms.is_empty() || slots.contains("action"));
    }
    let required_non_core = plan
        .required_slots
        .iter()
        .filter(|slot| {
            !matches!(
                slot.as_str(),
                "answer_bearing_fact" | "subject" | "temporal_scope" | "requested_modality"
            )
        })
        .collect::<Vec<_>>();
    !required_non_core.is_empty()
        && required_non_core
            .iter()
            .all(|slot| slots.contains(slot.as_str()))
}

fn promote_selected_evidence_chains(
    plan: &QueryPlan,
    atoms: &mut [EvidenceAtom],
    selected_atom_ids: &[String],
) {
    let selected_indexes = atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| {
            selected_atom_ids.contains(&atom.atom_id)
                && atom.kind.grounding_role() == GroundingRole::Original
                && atom_can_contribute_to_evidence_chain(plan, atom)
        })
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if selected_indexes.len() < 2 {
        return;
    }

    for root in &selected_indexes {
        let mut members = vec![*root];
        let mut relation_reasons = Vec::<String>::new();
        while members.len() < 4 {
            let next = selected_indexes
                .iter()
                .copied()
                .filter(|index| !members.contains(index))
                .filter_map(|index| {
                    let relation = members.iter().find_map(|member| {
                        let relation =
                            evidence_chain_relation(plan, &atoms[*member], &atoms[index]);
                        (!relation.is_empty()).then_some(relation)
                    })?;
                    let member_slots = members
                        .iter()
                        .flat_map(|member| evidence_chain_slots(plan, &atoms[*member]))
                        .collect::<BTreeSet<_>>();
                    let candidate_slots = evidence_chain_slots(plan, &atoms[index]);
                    let new_slots = candidate_slots
                        .iter()
                        .filter(|slot| !member_slots.contains(*slot))
                        .count();
                    Some((index, relation, new_slots, atoms[index].score))
                })
                .max_by(|left, right| {
                    left.2
                        .cmp(&right.2)
                        .then_with(|| left.3.partial_cmp(&right.3).unwrap_or(Ordering::Equal))
                });
            let Some((index, relation, _, _)) = next else {
                break;
            };
            relation_reasons.extend(relation);
            members.push(index);
            let combined_slots = members
                .iter()
                .flat_map(|member| evidence_chain_slots(plan, &atoms[*member]))
                .collect::<BTreeSet<_>>();
            if evidence_chain_covers_answer(plan, &combined_slots) {
                let primary = members
                    .iter()
                    .copied()
                    .filter(|index| {
                        let slots = evidence_chain_slots(plan, &atoms[*index]);
                        match plan.answer_target {
                            AnswerTarget::Location => slots.contains("location"),
                            AnswerTarget::Temporal => slots.contains("event_time"),
                            _ => slots.contains("object") || slots.contains("action"),
                        }
                    })
                    .max_by(|left, right| {
                        atoms[*left]
                            .score
                            .partial_cmp(&atoms[*right].score)
                            .unwrap_or(Ordering::Equal)
                    })
                    .unwrap_or(*root);
                for member in &members {
                    let chain_slots = evidence_chain_slots(plan, &atoms[*member]);
                    let mut slots = atoms[*member]
                        .supports_slots
                        .iter()
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    slots.extend(chain_slots);
                    slots.insert("evidence_chain_member".to_string());
                    for reason in &relation_reasons {
                        slots.insert(format!("evidence_chain_relation:{reason}"));
                    }
                    if *member == primary {
                        slots.insert("answer_bearing_fact".to_string());
                        slots.insert("evidence_chain_primary".to_string());
                    } else {
                        slots.insert("evidence_chain_support".to_string());
                    }
                    atoms[*member].supports_slots = slots.into_iter().collect();
                }
                return;
            }
        }
    }
}

fn expand_selected_evidence_atom_ids(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    selected_answer_atom_ids: &[String],
) -> HashSet<String> {
    let mut selected = selected_answer_atom_ids
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    if selected.is_empty() {
        return selected;
    }
    let primary_atoms = atoms
        .iter()
        .filter(|atom| {
            selected.contains(&atom.atom_id)
                && atom.kind.grounding_role() == GroundingRole::Original
        })
        .collect::<Vec<_>>();

    if primary_atoms
        .iter()
        .any(|atom| atom_covers_complete_bounded_list(plan, atom))
    {
        return selected;
    }

    // A text statement and an image/audio caption in one turn are one
    // evidentiary unit. Keep only answer-bearing siblings from that turn,
    // rather than reopening the whole record.
    for primary in &primary_atoms {
        let include_same_turn_decompression_context = plan.grounded_list_kind
            == Some(crate::memory_engine_summary::GroundedListKind::Destress)
            && primary
                .supports_slots
                .iter()
                .any(|slot| slot == "same_turn_list_value_bridge");
        for sibling in atoms.iter().filter(|candidate| {
            candidate.memory_id == primary.memory_id
                && candidate.context_group == primary.context_group
                && candidate.kind.grounding_role() == GroundingRole::Original
                && (atom_is_answer_bearing(candidate)
                    || (include_same_turn_decompression_context
                        && crate::memory_engine_summary::source_signals_decompression(
                            &candidate.text,
                        )))
        }) {
            selected.insert(sibling.atom_id.clone());
        }
    }
    if plan.grounded_list_kind == Some(crate::memory_engine_summary::GroundedListKind::Books)
        && primary_atoms
            .iter()
            .any(|atom| atom.supports_slots.iter().any(|slot| slot == "list_value"))
    {
        let mut unresolved_references = atoms
            .iter()
            .filter(|candidate| {
                !selected.contains(&candidate.atom_id)
                    && candidate.kind.grounding_role() == GroundingRole::Original
                    && candidate
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "list_unresolved_reference")
            })
            .collect::<Vec<_>>();
        unresolved_references.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.memory_id.cmp(&right.memory_id))
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        });
        for candidate in unresolved_references.into_iter().take(2) {
            selected.insert(candidate.atom_id.clone());
        }
    }

    if plan.media_provenance_requested {
        for primary in primary_atoms.iter().filter(|primary| {
            primary
                .supports_slots
                .iter()
                .any(|slot| slot == "media_provenance_bridge")
        }) {
            for media in atoms.iter().filter(|candidate| {
                candidate.memory_id == primary.memory_id
                    && candidate.kind == EvidenceKind::NativeMedia
                    && candidate.context_group != usize::MAX
                    && primary.context_group != usize::MAX
                    && candidate.context_group.abs_diff(primary.context_group) <= 4
            }) {
                selected.insert(media.atom_id.clone());
            }
        }
    }

    if plan.participant_relationship_qualifier {
        for primary in primary_atoms.iter().filter(|primary| {
            primary
                .supports_slots
                .iter()
                .any(|slot| slot == "relationship_actor_conflict")
        }) {
            let Some(speaker) = primary.speaker.as_deref() else {
                continue;
            };
            for context in atoms.iter().filter(|candidate| {
                candidate.memory_id == primary.memory_id
                    && candidate.ordinal <= primary.ordinal
                    && primary.ordinal.saturating_sub(candidate.ordinal) <= 4
                    && candidate.kind.grounding_role() == GroundingRole::Original
                    && candidate
                        .speaker
                        .as_deref()
                        .is_some_and(|candidate_speaker| {
                            normalize_person_key(candidate_speaker) == normalize_person_key(speaker)
                        })
                    && plan
                        .qualifier_terms
                        .iter()
                        .all(|term| text_matches_lexical_term(&candidate.text.to_lowercase(), term))
            }) {
                selected.insert(context.atom_id.clone());
            }
        }
    }

    // A source-level qualifier bridge is valid only because a separate,
    // same-subject sentence carries the complete qualifier. Keep that sentence
    // in the final evidence plan rather than attributing its slots to the
    // answer sentence itself.
    if source_level_qualifier_bridge_allowed(plan) {
        for primary in primary_atoms.iter().filter(|primary| {
            primary
                .supports_slots
                .iter()
                .any(|slot| slot == "source_qualifier_bridge")
        }) {
            for context in atoms.iter().filter(|candidate| {
                candidate.memory_id == primary.memory_id
                    && candidate
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "source_qualifier_anchor")
            }) {
                selected.insert(context.atom_id.clone());
            }
        }
    }

    // A manner/description answer can be expressed by a nearby follow-up from
    // the same speaker while the requested activity appears in the preceding
    // turn. Keep at most two such complementary facts, not the full dialogue.
    if !plan.qualifier_terms.is_empty()
        && matches!(
            plan.answer_target,
            AnswerTarget::Manner | AnswerTarget::Description
        )
    {
        let mut complementary = atoms
            .iter()
            .filter(|candidate| {
                !selected.contains(&candidate.atom_id)
                    && candidate.kind.grounding_role() == GroundingRole::Original
                    && candidate.context_group != usize::MAX
                    && !is_low_information_statement(&candidate.text)
                    && primary_atoms.iter().any(|primary| {
                        primary.memory_id == candidate.memory_id
                            && primary.context_group != usize::MAX
                            && primary.context_group.abs_diff(candidate.context_group) <= 4
                            && primary.speaker.is_some()
                            && primary.speaker == candidate.speaker
                    })
            })
            .collect::<Vec<_>>();
        complementary.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.ordinal.cmp(&right.ordinal))
        });
        for candidate in complementary.into_iter().take(2) {
            selected.insert(candidate.atom_id.clone());
        }
    }

    let temporal_memory_ids = atoms
        .iter()
        .filter(|atom| {
            selected.contains(&atom.atom_id)
                && atom.kind.grounding_role() == GroundingRole::Original
                && (plan.answer_target == AnswerTarget::Temporal || plan.temporal_scope != "any")
        })
        .map(|atom| atom.memory_id)
        .collect::<HashSet<_>>();
    for memory_id in temporal_memory_ids {
        if let Some(session_date) = atoms
            .iter()
            .find(|atom| atom.memory_id == memory_id && atom.kind == EvidenceKind::SessionDate)
        {
            selected.insert(session_date.atom_id.clone());
        }
    }
    selected
}

fn selected_memory_ids_for_atoms(atom_ids: &[String], atoms: &[EvidenceAtom]) -> Vec<u64> {
    let mut seen = HashSet::new();
    atom_ids
        .iter()
        .filter_map(|atom_id| {
            atoms
                .iter()
                .find(|atom| atom.atom_id == *atom_id)
                .map(|atom| atom.memory_id)
        })
        .filter(|memory_id| seen.insert(*memory_id))
        .collect()
}

fn claim_evidence_bindings_from_atoms(
    atoms: &[EvidenceAtom],
    evidence_ids: &[u64],
) -> (Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>) {
    let memory_ids = evidence_ids.iter().copied().collect::<HashSet<_>>();
    let mut selected = atoms
        .iter()
        .filter(|atom| memory_ids.contains(&atom.memory_id))
        .filter(|atom| {
            atom.kind.grounding_role() == GroundingRole::Original
                && (atom_is_answer_bearing(atom)
                    || atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "evidence_chain_support"))
        })
        .collect::<Vec<_>>();
    if selected.is_empty() {
        selected = atoms
            .iter()
            .filter(|atom| {
                memory_ids.contains(&atom.memory_id)
                    && atom.kind.grounding_role() == GroundingRole::Original
            })
            .collect();
    }
    selected.sort_by(|left, right| {
        left.memory_id
            .cmp(&right.memory_id)
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
    let evidence_atom_ids = selected
        .iter()
        .map(|atom| atom.atom_id.clone())
        .collect::<Vec<_>>();
    let mut anchor_keys = HashSet::new();
    let evidence_source_anchors = selected
        .iter()
        .filter_map(|atom| atom.source_anchor.clone())
        .filter(|anchor| anchor_keys.insert(format!("{}:{}", anchor.kind, anchor.value)))
        .collect::<Vec<_>>();
    let mut derivation = selected
        .iter()
        .flat_map(|atom| atom.supports_slots.iter())
        .filter_map(|slot| {
            slot.strip_prefix("evidence_chain_relation:")
                .map(str::to_string)
                .or_else(|| {
                    (slot == "evidence_chain_support")
                        .then(|| "complementary_evidence_chain".to_string())
                })
        })
        .collect::<Vec<_>>();
    if evidence_ids.len() > 1 && derivation.is_empty() {
        derivation.push("multiple_source_evidence".to_string());
    }
    derivation.sort();
    derivation.dedup();
    (evidence_atom_ids, evidence_source_anchors, derivation)
}

fn claim_evidence_bindings_for_atom_ids(
    atoms: &[EvidenceAtom],
    atom_ids: &[String],
) -> (Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>) {
    let wanted = atom_ids.iter().map(String::as_str).collect::<HashSet<_>>();
    let mut selected = atoms
        .iter()
        .filter(|atom| wanted.contains(atom.atom_id.as_str()))
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    selected.sort_by(|left, right| {
        left.memory_id
            .cmp(&right.memory_id)
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
    claim_evidence_binding_fields(&selected)
}



fn claim_explicitly_disavows_subject(plan: &QueryPlan, claim_lower: &str) -> bool {
    let Some(subject) = plan.subject.as_deref() else {
        return false;
    };
    let subject = subject.to_lowercase();
    [
        format!("not {subject}"),
        format!("不是{subject}"),
        format!("不是 {subject}"),
        format!("并非{subject}"),
        format!("并非 {subject}"),
    ]
    .iter()
    .any(|pattern| claim_lower.contains(pattern))
}

fn answer_claim_content(claim_without_citations: &str) -> &str {
    let trimmed = claim_without_citations.trim_start();
    let lower = trimmed.to_ascii_lowercase();
    [
        "direct answer:",
        "partial answer:",
        "likely answer:",
        "inferred answer:",
        "answer:",
        "直接回答：",
        "部分回答：",
        "可能回答：",
    ]
    .iter()
    .find_map(|prefix| lower.starts_with(prefix).then(|| &trimmed[prefix.len()..]))
    .unwrap_or(trimmed)
    .trim_start()
}

fn claim_remainder_starts_with_subject_predicate(plan: &QueryPlan, remainder: &str) -> bool {
    let mut calendar_words = word_regex().find_iter(remainder);
    if calendar_words
        .next()
        .is_some_and(|word| word.as_str().eq_ignore_ascii_case("may"))
        && calendar_words.next().is_some_and(|word| {
            word.as_str()
                .parse::<u32>()
                .is_ok_and(|day| (1..=31).contains(&day))
        })
    {
        return false;
    }
    let predicate = word_regex()
        .find_iter(remainder)
        .map(|value| value.as_str().to_lowercase())
        .skip_while(|word| claim_predicate_modifier(word))
        .next()
        .unwrap_or_default();
    if matches!(
        predicate.as_str(),
        "am" | "are"
            | "be"
            | "been"
            | "being"
            | "did"
            | "do"
            | "does"
            | "had"
            | "has"
            | "have"
            | "is"
            | "was"
            | "were"
            | "will"
            | "would"
            | "can"
            | "could"
            | "may"
            | "might"
            | "must"
            | "need"
            | "needs"
            | "should"
            | "seems"
            | "appears"
    ) {
        return true;
    }
    plan.action_terms
        .iter()
        .any(|term| text_matches_term(&predicate, term))
}

fn claim_predicate_modifier(word: &str) -> bool {
    // Closed lexical class: a name ending in `ly` must not be discarded as an
    // adverb. Unsupported modifiers remain unverified rather than guessed.
    matches!(
        word,
        "carefully"
            | "quickly"
            | "slowly"
            | "quietly"
            | "thoroughly"
            | "successfully"
            | "recently"
            | "previously"
            | "already"
            | "also"
            | "really"
            | "very"
            | "quite"
            | "often"
            | "always"
            | "usually"
    )
}

fn count_claim_is_scalar_or_owned_value(content: &str, subject: &str) -> bool {
    let lower = content.to_lowercase();
    let scalar = source_condition_parts(&lower)
        .map(|(head, _, _, _)| head)
        .unwrap_or(&lower);
    let owner = format!("{}'s ", subject.to_lowercase());
    let curly_owner = format!("{}\u{2019}s ", subject.to_lowercase());
    let value = scalar
        .strip_prefix(&owner)
        .or_else(|| lower.strip_prefix(&curly_owner))
        .unwrap_or(scalar);
    static COUNT_HEADING: OnceLock<Regex> = OnceLock::new();
    let heading = COUNT_HEADING.get_or_init(|| {
        Regex::new(r"^(?:the )?(?:total|count|number of times)(?:\s+is|\s*:)?\s*")
            .expect("owned count heading regex")
    });
    let value = heading
        .find(value)
        .map_or(value, |prefix| &value[prefix.end()..]);
    let value = value
        .trim()
        .trim_end_matches(['.', '。', '!', '！'])
        .trim_end();
    if !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()) {
        return value.parse::<usize>().is_ok();
    }
    // Use the renderer's quantity vocabulary, but require the entire scalar
    // to match: a number inside an unrelated person's statement is not enough.
    explicit_count_value_with_number(value).is_some_and(|(_, display)| display == value)
}

fn claim_postposed_agents_match(plan: &QueryPlan, content: &str, expected: &str) -> bool {
    let lower = content.to_lowercase();
    for by in word_regex()
        .find_iter(&lower)
        .filter(|word| word.as_str() == "by")
    {
        let clause = &lower;
        if !plan
            .action_terms
            .iter()
            .any(|term| text_matches_term(clause, term))
        {
            continue;
        }
        let tail = lower[by.end()..]
            .split([',', '.', ';', '!', '?', '\n'])
            .next()
            .unwrap_or_default();
        let tail = tail.trim();
        if date_regex()
            .find(tail)
            .is_some_and(|date| date.start() == 0)
            || contains_bare_calendar_month_reference(&format!("by {tail}"))
            || ["tomorrow", "yesterday", "today", "noon", "midnight"]
                .iter()
                .any(|value| contains_bounded_ascii_phrase(tail, value))
        {
            continue;
        }
        let agent = word_regex()
            .find_iter(tail)
            .map(|word| word.as_str())
            .take_while(|word| {
                !matches!(
                    *word,
                    "on" | "at" | "in" | "before" | "after" | "during" | "when" | "if" | "using"
                ) && !claim_predicate_modifier(word)
            })
            .collect::<Vec<_>>()
            .join(" ");
        // Explicit but unresolved agents fail closed; they cannot inherit the
        // question subject like a bare numerical answer can.
        if agent.is_empty() || normalize_person_key(&agent) != normalize_person_key(expected) {
            return false;
        }
    }
    true
}

fn claim_subject_matches_plan(plan: &QueryPlan, claim_without_citations: &str) -> bool {
    let Some(expected_subject) = plan.subject.as_deref() else {
        return true;
    };
    let content = answer_claim_content(claim_without_citations);
    if !claim_postposed_agents_match(plan, content, expected_subject) {
        return false;
    }
    if plan.answer_target == AnswerTarget::Count
        && !word_regex().find_iter(content).any(|token| {
            plan.action_terms
                .iter()
                .any(|action| text_matches_term(&token.as_str().to_lowercase(), action))
        })
    {
        return count_claim_is_scalar_or_owned_value(content, expected_subject);
    }
    if plan.answer_target == AnswerTarget::Temporal {
        static DATE_VALUE_CLAIM: OnceLock<Regex> = OnceLock::new();
        if DATE_VALUE_CLAIM.get_or_init(|| Regex::new(
            // Bare date values rendered by the event-date operator inherit the
            // question's subject; the temporal proof still binds the value.
            r"(?i)^(?:the date was |日期是\s*)(?:(?:the week of |the (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday) before )?(?:[0-9]{1,2} )?(?:(?:january|february|march|april|may|june|july|august|september|october|november|december) )?[0-9]{4}|(?:spring|summer|autumn|fall|winter) [0-9]{4}|[0-9]{4}年[^。.]*)[。. ]*$"
        ).expect("scalar date claim regex")).is_match(content) {
            return true;
        }
    }
    // "Fitting into the new team was a challenge": a leading gerund phrase has
    // no explicit actor, so it inherits the question's subject; the bound atom
    // must still be attributed to that subject.
    let trimmed = content.trim_start();
    let first_word_end = trimmed
        .find(|ch: char| !ch.is_alphanumeric() && ch != '\'')
        .unwrap_or(trimmed.len());
    if sentence_initial_gerund(trimmed, 0, first_word_end) {
        return true;
    }
    // An explicit actor can occur after an adjunct or in a later clause.
    // Bare answer values inherit the question's subject, not arbitrary names.
    for token in word_regex().find_iter(content) {
        if claim_predicate_modifier(&token.as_str().to_lowercase()) {
            continue;
        }
        let remainder = content[token.end()..].trim_start();
        if claim_remainder_starts_with_subject_predicate(plan, remainder) {
            let subject = normalize_person_key(token.as_str());
            if !matches!(
                subject.as_str(),
                "not"
                    | "never"
                    | "did"
                    | "does"
                    | "do"
                    | "has"
                    | "have"
                    | "had"
                    | "was"
                    | "were"
                    | "is"
                    | "are"
                    | "will"
                    | "would"
                    | "can"
                    | "could"
                    | "should"
                    | "must"
                    | "may"
                    | "might"
            ) && subject != normalize_person_key(expected_subject)
            {
                return false;
            }
        }
    }
    true
}

/// A partial answer may state that the evidence belongs to a different actor,
/// but only when the final Claim quotes that exact, durably-attributed fact and
/// explicitly disavows the requested subject. This is not normal subject
/// matching and cannot be reached from an arbitrary name mention.
fn attribution_conflict_claim_bindings(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    claim_lower: &str,
    memory_ids: &HashSet<u64>,
) -> Option<(Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>)> {
    if !claim_explicitly_disavows_subject(plan, claim_lower) {
        return None;
    }
    let requested_subject = plan.subject.as_deref()?;
    let answer_atom = atoms.iter().find(|atom| {
        if !memory_ids.contains(&atom.memory_id)
            || !atom_is_answer_bearing(atom)
            || atom.kind.grounding_role() != GroundingRole::Original
        {
            return false;
        }
        let relationship_conflict = atom
            .supports_slots
            .iter()
            .any(|slot| slot == "relationship_actor_conflict");
        let collective_conflict = atom
            .supports_slots
            .iter()
            .any(|slot| slot == "collective_participant_context_bridge");
        if !(collective_conflict
            || (plan.participant_relationship_qualifier && relationship_conflict))
        {
            return false;
        }
        let Some(actor) = atom.speaker.as_deref() else {
            return false;
        };
        normalize_person_key(actor) != normalize_person_key(requested_subject)
            && atom_has_verified_fact_subject(atom, actor)
            && claim_lower.contains(&actor.to_lowercase())
            && claim_lower.contains(&clean_answer_statement(&atom.text).to_lowercase())
    })?;
    let (atom_ids, anchors, mut derivation) = claim_evidence_binding_fields(&[answer_atom]);
    derivation.push("attribution_conflict:requested_subject_mismatch".to_string());
    derivation.sort();
    derivation.dedup();
    Some((atom_ids, anchors, derivation))
}

fn claim_contains_temporal_value(claim_lower: &str, english: &str, chinese: &str) -> bool {
    claim_lower.contains(&english.to_lowercase()) || claim_lower.contains(&chinese.to_lowercase())
}

/// Lexical fallback for event dates without a confident semantic score.
/// "Take a trip" / "have a party": when every question predicate is a light
/// verb, the event is named by its object, so an atom stating the object
/// ("my Rocky Mountains trip last year") covers the predicate.
const LIGHT_VERBS: &[&str] = &[
    "take", "took", "taken", "taking", "have", "had", "having", "get", "got", "gotten",
    "getting", "make", "made", "making", "do", "did", "done", "doing", "go", "went", "gone",
    "going",
];

fn light_verb_predicate_covered_by_object(plan: &QueryPlan, atom: &EvidenceAtom) -> bool {
    let text = atom.text.to_lowercase();
    !plan.action_terms.is_empty()
        && plan
            .action_terms
            .iter()
            .all(|term| LIGHT_VERBS.contains(&term.to_lowercase().as_str()))
        && plan
            .object_terms
            .iter()
            .any(|term| text_matches_term(&text, term))
}

/// A bare date Claim carries no event content, so the dated atom must match
/// the question's actors. "When did Jolene's parents give her a console?" is
/// not answered by Jolene buying her partner one: every possessed relation
/// actor the question names must occur in the event's turn.
fn temporal_event_covers_relation_terms(plan: &QueryPlan, same_turn: &[&EvidenceAtom]) -> bool {
    plan.possessive_relation_actors.iter().all(|term| {
        same_turn
            .iter()
            .any(|atom| text_matches_term(&atom.text.to_lowercase(), term))
    })
}

/// "When did James start taking cooking classes?" is not answered by James
/// starting a road trip: when the question names an object, the event's turn
/// must state at least one of its content words (light verbs do not count).
fn temporal_event_covers_object_terms(plan: &QueryPlan, same_turn: &[&EvidenceAtom]) -> bool {
    let content_terms = plan
        .object_terms
        .iter()
        .filter(|term| !LIGHT_VERBS.contains(&term.to_lowercase().as_str()))
        .collect::<Vec<_>>();
    content_terms.is_empty()
        || content_terms.iter().any(|term| {
            same_turn
                .iter()
                .any(|atom| text_matches_term(&atom.text.to_lowercase(), term))
        })
}

fn temporal_claim_bindings(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    claim_lower: &str,
    memory_ids: &HashSet<u64>,
) -> Option<(Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>)> {
    if plan.answer_target != AnswerTarget::Temporal {
        return None;
    }
    let requested_years = plan
        .time_terms
        .iter()
        .filter_map(|term| {
            (term.len() == 4 && term.chars().all(|character| character.is_ascii_digit()))
                .then(|| term.parse::<i32>().ok())
                .flatten()
        })
        .collect::<HashSet<_>>();

    for event in atoms.iter().filter(|atom| {
        memory_ids.contains(&atom.memory_id)
            && atom_is_answer_bearing(atom)
            && atom.kind.grounding_role() == GroundingRole::Original
    }) {
        // A turn is often split into several atoms ("have you been to Italy?
        // I had a blast there last month"); a named entity may be stated in
        // another sentence of the same turn.
        let same_turn = atoms
            .iter()
            .filter(|atom| {
                atom.memory_id == event.memory_id
                    && atom.context_group == event.context_group
                    && atom.kind.grounding_role() == GroundingRole::Original
            })
            .collect::<Vec<_>>();
        if plan
            .subject
            .as_deref()
            .is_some_and(|subject| !atom_has_verified_fact_subject(event, subject))
            || !semantic_coverage(&[event]).unwrap_or_else(|| {
                (claim_atoms_cover_query_predicate(plan, &[event])
                    || light_verb_predicate_covered_by_object(plan, event))
                    && temporal_event_covers_object_terms(plan, &same_turn)
            })
            || !plan.named_entities.iter().all(|entity| {
                same_turn
                    .iter()
                    .any(|atom| text_matches_named_entity(&atom.text.to_lowercase(), entity))
            })
            || !temporal_event_covers_relation_terms(plan, &same_turn)
        {
            continue;
        }
        let Some(session_date_atom) = atoms.iter().find(|candidate| {
            candidate.memory_id == event.memory_id && candidate.kind == EvidenceKind::SessionDate
        }) else {
            continue;
        };

        if let Some(session_date) =
            crate::memory_engine_summary::source_session_date(&session_date_atom.text)
        {
            let Some((english, chinese, marker)) =
                crate::memory_engine_summary::relative_event_date_answer(&event.text, session_date)
            else {
                continue;
            };
            let year = session_date.format("%Y").to_string().parse::<i32>().ok();
            if (!requested_years.is_empty()
                && !year.is_some_and(|value| requested_years.contains(&value)))
                || !claim_contains_temporal_value(claim_lower, &english, &chinese)
            {
                continue;
            }
            let (atom_ids, anchors, mut derivation) =
                claim_evidence_binding_fields(&[event, session_date_atom]);
            derivation.push("same_memory_session_date".to_string());
            derivation.push(format!("relative_event_date:{marker}"));
            derivation.sort();
            derivation.dedup();
            return Some((atom_ids, anchors, derivation));
        }

        let Some(month_year) = source_session_month_year(&session_date_atom.text) else {
            continue;
        };
        if !temporal_granularity_accepts_month_year(plan.temporal_granularity)
            || has_unresolved_temporal_reference(&event.text)
        {
            continue;
        }
        let year = number_regex()
            .find_iter(&month_year)
            .find_map(|value| value.as_str().parse::<i32>().ok());
        if (!requested_years.is_empty()
            && !year.is_some_and(|value| requested_years.contains(&value)))
            || !claim_lower.contains(&month_year.to_lowercase())
        {
            continue;
        }
        let (atom_ids, anchors, mut derivation) =
            claim_evidence_binding_fields(&[event, session_date_atom]);
        derivation.push("same_memory_session_month_year".to_string());
        derivation.sort();
        derivation.dedup();
        return Some((atom_ids, anchors, derivation));
    }
    None
}

/// A multi-entity partial may prove a premise without proving the requested
/// outcome. Bind that premise only when its rendered statement and its one
/// explicitly named entity both occur in the final Claim.
fn inference_premise_claim_bindings(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    claim_lower: &str,
    memory_ids: &HashSet<u64>,
) -> Option<(Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>)> {
    if !is_multi_entity_inference(plan) {
        return None;
    }
    let subject = plan.subject.as_deref()?;
    let premise = atoms.iter().find(|atom| {
        memory_ids.contains(&atom.memory_id)
            && atom_is_inference_entity_context(atom)
            && atom_has_verified_fact_subject(atom, subject)
            && plan.named_entities.iter().any(|entity| {
                atom.supports_slots
                    .contains(&named_entity_slot_name(entity))
                    && text_matches_named_entity(claim_lower, entity)
            })
            // This branch certifies only the premise, not a larger inference
            // that happens to quote it. Extra conclusions need their own proof.
            && claim_lower.trim() == render_answer_statement(plan, atom).to_lowercase().trim()
    })?;
    let (atom_ids, anchors, mut derivation) = claim_evidence_binding_fields(&[premise]);
    derivation.push("inference_premise_context".to_string());
    derivation.sort();
    derivation.dedup();
    Some((atom_ids, anchors, derivation))
}

fn scoped_negation_condition_claim_bindings(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    claim_without_citations: &str,
    claim_lower: &str,
    memory_ids: &HashSet<u64>,
) -> Option<(Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>)> {
    if plan.negation_terms.is_empty()
        || plan.condition_terms.is_empty()
        || extract_negation_terms(claim_lower).is_empty()
    {
        return None;
    }
    let subject_terms = plan
        .subject
        .as_deref()
        .map(|subject| vec![subject.to_lowercase()])
        .unwrap_or_default();
    let claim_content =
        claim_content_terms(&informative_terms(claim_without_citations), &subject_terms);
    let claim_values = claim_answer_value_terms(plan, &claim_content);
    let proof_terms = if claim_values.is_empty() {
        &claim_content
    } else {
        &claim_values
    };
    let fact = atoms.iter().find(|atom| {
        let lower = atom.text.to_lowercase();
        memory_ids.contains(&atom.memory_id)
            && atom_is_answer_bearing(atom)
            && atom.kind.grounding_role() == GroundingRole::Original
            && plan
                .subject
                .as_deref()
                .is_none_or(|subject| atom_has_verified_fact_subject(atom, subject))
            && claim_atoms_cover_query_predicate(plan, &[atom])
            && negation_scope_matches_plan(plan, &lower, proof_terms)
            && strict_condition_terms_match_evidence(plan, &lower)
            && proof_terms
                .iter()
                .all(|term| text_matches_term(&lower, term))
    })?;
    let (atom_ids, anchors, mut derivation) = claim_evidence_binding_fields(&[fact]);
    derivation.push("scoped_negation_condition".to_string());
    derivation.sort();
    derivation.dedup();
    Some((atom_ids, anchors, derivation))
}

fn verified_claim_bindings_for_plan(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    claim: &str,
    evidence_ids: &[u64],
) -> (Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>) {
    let bindings = proposed_claim_bindings_for_plan(plan, atoms, claim, evidence_ids);
    if bindings.2.iter().any(|step| step == "date_difference_v1") {
        return bindings;
    }
    let claim_without_citations = citation_regex().replace_all(claim, " ");
    let lower = claim_without_citations.to_lowercase();
    // Numerical and temporal branches must pass the same final logic check.
    if atoms.iter().any(|atom| {
        bindings.0.contains(&atom.atom_id)
            && atom.kind.grounding_role() == GroundingRole::Original
            // Typed profile modality observations are sibling provenance for
            // the exact summary Claim, not alternate wording of that Claim.
            // Applying condition/negation equivalence to them would reject a
            // valid summary merely because it contains words such as
            // "when", "without", or "one-off" that the descriptive
            // observation does not repeat. The summary atom itself still
            // passes the full logic check below.
            && !(plan_requests_structured_profile(plan)
                && atom.subject_binding.iter().any(|binding| {
                    binding == "structured_profile_modality_observation"
                }))
            // A typed profile summary Claim is an exact quotation of the
            // original summary atom. Its own negation/condition wording is
            // therefore already source-equal and must not be rejected merely
            // because the user question did not repeat that wording.
            && !(plan_requests_structured_profile(plan)
                && atom
                    .subject_binding
                    .iter()
                    .any(|binding| binding == "structured_profile_summary")
                && normalize_for_dedup(answer_claim_content(&claim_without_citations))
                    == normalize_for_dedup(&clean_answer_statement(&atom.text)))
            && !claim_logic_matches_source(
                plan,
                if bindings
                    .2
                    .iter()
                    .any(|step| step == "attribution_conflict:requested_subject_mismatch")
                {
                    // The disavowal is about attribution, not negation of the
                    // quoted event. Its exact quotation was checked above.
                    &atom.text
                } else {
                    &lower
                },
                &atom.text.to_lowercase(),
            )
    }) {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    bindings
}

fn claim_logic_matches_source(plan: &QueryPlan, claim: &str, source: &str) -> bool {
    match (
        claim_event_condition_clause(plan, claim),
        claim_event_condition_clause(plan, source),
    ) {
        (Some((left, left_clause)), Some((right, right_clause))) => {
            if left != right
                || extract_negation_terms(left_clause) != extract_negation_terms(right_clause)
                || ordered_condition_terms(left_clause) != ordered_condition_terms(right_clause)
            {
                return false;
            }
        }
        (None, None) => {}
        // A reason or time clause the claim omits ("since we last spoke",
        // "because the sensor failed") does not make the source event
        // hypothetical; a condition (if/unless/when/only...) would.
        (None, Some((ConditionRelation::Causal, _))) => {}
        _ => return false,
    }
    let values = claim_answer_value_terms(
        plan,
        &claim_content_terms(
            &informative_terms(claim),
            &plan
                .subject
                .iter()
                .map(|s| s.to_lowercase())
                .collect::<Vec<_>>(),
        ),
    );
    let claim_negated = negation_scope_matches_plan(plan, claim, &values);
    let source_negated = negation_scope_matches_plan(plan, source, &values);
    // Failure to resolve a negative clause is not proof of a positive fact.
    if (!extract_negation_terms(claim).is_empty() && !claim_negated)
        || (!extract_negation_terms(source).is_empty() && !source_negated)
    {
        return false;
    }
    claim_negated == source_negated
}

fn claim_event_condition_clause<'a>(
    plan: &QueryPlan,
    value: &'a str,
) -> Option<(ConditionRelation, &'a str)> {
    let (relation, tail) = source_condition_clause(value)?;
    if relation == ConditionRelation::When && !plan.action_terms.is_empty() {
        let before = &value[..value.len() - tail.len()];
        let has_action = |text: &str| {
            plan.action_terms
                .iter()
                .any(|action| text_matches_term(text, action))
        };
        // "A challenge when I hurt my ankle" contains the requested event
        // inside the when-clause; it does not condition an earlier injury.
        let has_main_clause = before
            .trim_end()
            .strip_suffix("when")
            .is_some_and(|prefix| !prefix.trim().is_empty());
        if has_main_clause && !has_action(before) && has_action(tail) {
            return None;
        }
    }
    Some((relation, tail))
}

fn ordered_condition_terms(clause: &str) -> Vec<String> {
    // Preserve roles, negation and connectives. A bag of words cannot
    // distinguish "sensor disables motor" from "motor disables sensor".
    word_regex()
        .find_iter(clause)
        .map(|word| word.as_str().to_lowercase())
        .collect()
}

fn proposed_claim_bindings_for_plan(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    claim: &str,
    evidence_ids: &[u64],
) -> (Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>) {
    let memory_ids = evidence_ids.iter().copied().collect::<HashSet<_>>();
    let claim_without_citations = citation_regex().replace_all(claim, " ");
    let claim_lower = claim_without_citations.to_lowercase();
    if let Some(proof) = duration_proof(plan, atoms) {
        for chinese in [false, true] {
            for (text, ids) in proof.claims(chinese) {
                if claim_without_citations
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase()
                    == text
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ")
                        .to_lowercase()
                    && selected_memory_ids_for_atoms(&ids, atoms)
                        .iter()
                        .all(|id| memory_ids.contains(id))
                {
                    let (ids, anchors, mut derivation) =
                        claim_evidence_bindings_for_atom_ids(atoms, &ids);
                    derivation.push("date_difference_v1".into());
                    derivation.push(format!(
                        "date_operands:{}:{}",
                        proof.first_date, proof.second_date
                    ));
                    return (ids, anchors, derivation);
                }
            }
        }
        return (Vec::new(), Vec::new(), Vec::new());
    }
    if plan.duration_clauses.is_some() {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    let exact_attributed_quote = plan.subject.as_deref().is_some_and(|subject| {
        atoms.iter().any(|atom| {
            memory_ids.contains(&atom.memory_id)
                && atom.kind.grounding_role() == GroundingRole::Original
                && atom_has_verified_fact_subject(atom, subject)
                && normalize_for_dedup(answer_claim_content(&claim_without_citations))
                    == normalize_for_dedup(&clean_answer_statement(&atom.text))
        })
    });
    if !exact_attributed_quote && !claim_subject_matches_plan(plan, &claim_without_citations) {
        if let Some(bindings) =
            attribution_conflict_claim_bindings(plan, atoms, &claim_lower, &memory_ids)
        {
            return bindings;
        }
        return (Vec::new(), Vec::new(), Vec::new());
    }
    if let Some(bindings) =
        attribution_conflict_claim_bindings(plan, atoms, &claim_lower, &memory_ids)
    {
        return bindings;
    }
    if plan.answer_target == AnswerTarget::Count {
        // A renderer cannot redefine an aggregate's scope by citing a subset.
        if !count_claim_is_scalar_or_owned_value(
            answer_claim_content(&claim_without_citations),
            plan.subject.as_deref().unwrap_or_default(),
        ) && !plan
            .action_terms
            .iter()
            .any(|action| text_matches_term(&claim_lower, action))
        {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        let count_events = count_event_set(plan, atoms, None);
        let Some(claim_value) = count_value_from_claim(&claim_without_citations) else {
            return (Vec::new(), Vec::new(), Vec::new());
        };
        if !count_events.is_proven()
            || count_events.numeric_value != claim_value
            || !count_events
                .memory_ids
                .iter()
                .all(|id| memory_ids.contains(id))
        {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        let (atom_ids, anchors, mut derivation) =
            claim_evidence_bindings_for_atom_ids(atoms, &count_events.atom_ids);
        derivation.extend(count_events.derivation);
        derivation.sort();
        derivation.dedup();
        return (atom_ids, anchors, derivation);
    }
    if let Some(bindings) = temporal_claim_bindings(plan, atoms, &claim_lower, &memory_ids) {
        return bindings;
    }
    if let Some(bindings) = inference_premise_claim_bindings(plan, atoms, &claim_lower, &memory_ids)
    {
        return bindings;
    }
    if let Some(bindings) = scoped_negation_condition_claim_bindings(
        plan,
        atoms,
        &claim_without_citations,
        &claim_lower,
        &memory_ids,
    ) {
        return bindings;
    }
    if let Some(bindings) =
        structured_field_claim_bindings(atoms, &claim_without_citations, &memory_ids)
    {
        return bindings;
    }
    let claim_terms = informative_terms(&claim_without_citations);
    let subject_terms = plan
        .subject
        .as_deref()
        .map(|subject| vec![subject.to_lowercase()])
        .unwrap_or_default();
    let claim_content_terms = claim_content_terms(&claim_terms, &subject_terms);
    let exact_typed_profile_summary = plan_requests_structured_profile(plan)
        && atoms.iter().any(|atom| {
            memory_ids.contains(&atom.memory_id)
                && atom.kind.grounding_role() == GroundingRole::Original
                && atom
                    .subject_binding
                    .iter()
                    .any(|binding| binding == "structured_profile_summary")
                && normalize_for_dedup(answer_claim_content(&claim_without_citations))
                    == normalize_for_dedup(&clean_answer_statement(&atom.text))
        });
    // A shared question predicate (for example `built`) proves neither the
    // answer value nor the relation to that value. Prefer the Claim's value
    // terms (`greenhouse`, `shed`, dates, etc.) once the planned action is
    // removed, so an adjacent same-record fact cannot satisfy a wrong answer.
    let claim_value_terms = claim_answer_value_terms(plan, &claim_content_terms);
    let proof_content_terms = if claim_value_terms.is_empty() {
        &claim_content_terms
    } else {
        &claim_value_terms
    };
    let claim_numbers = number_regex()
        .find_iter(&claim_lower)
        .map(|value| value.as_str().to_string())
        .collect::<BTreeSet<_>>();
    let claim_negated = !extract_negation_terms(&claim_lower).is_empty();

    let mut selected = atoms
        .iter()
        .filter(|atom| {
            memory_ids.contains(&atom.memory_id)
                && atom.kind.grounding_role() == GroundingRole::Original
        })
        .filter(|atom| {
            let explicit_actor_conflict = plan.subject.as_deref().is_some_and(|subject| {
                claim_lower.contains(&format!("not {}", subject.to_lowercase()))
                    && atom.supports_slots.iter().any(|slot| {
                        matches!(
                            slot.as_str(),
                            "relationship_actor_conflict" | "collective_participant_context_bridge"
                        )
                    })
            });
            plan.subject.as_deref().is_none_or(|subject| {
                atom_has_verified_fact_subject(atom, subject) || explicit_actor_conflict
            })
        })
        .filter(|atom| {
            atom_is_answer_bearing(atom)
                || (exact_attributed_quote
                    && atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "list_unresolved_reference"))
                || atom.supports_slots.iter().any(|slot| {
                    matches!(
                        slot.as_str(),
                        "inference_entity_context"
                            | "qualifier_context"
                            | "source_qualifier_anchor"
                    )
                })
        })
        .filter(|atom| {
            let lower = atom.text.to_lowercase();
            if plan.achievement_extremum
                && !atom
                    .supports_slots
                    .iter()
                    .any(|slot| slot == "achievement_extremum")
            {
                return false;
            }
            if !claim_numbers.is_empty()
                && plan.answer_target != AnswerTarget::Count
                && !numeric_values_supported(&claim_without_citations, &atom.text)
            {
                return false;
            }
            if claim_negated
                && !exact_typed_profile_summary
                && !negation_scope_matches_plan(plan, &lower, proof_content_terms)
            {
                return false;
            }
            if claim_content_terms.is_empty() {
                return matches!(
                    plan.answer_target,
                    AnswerTarget::Location | AnswerTarget::Temporal
                );
            }
            proof_content_terms
                .iter()
                .all(|term| text_matches_term(&lower, term))
        })
        .collect::<Vec<_>>();

    if selected.is_empty() {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let relation_keys = selected
        .iter()
        .filter(|atom| {
            atom.supports_slots
                .iter()
                .any(|slot| slot == "evidence_chain_primary")
        })
        .flat_map(|atom| atom.supports_slots.iter())
        .filter_map(|slot| slot.strip_prefix("evidence_chain_relation:"))
        .collect::<HashSet<_>>();
    if !relation_keys.is_empty() {
        selected.extend(atoms.iter().filter(|candidate| {
            memory_ids.contains(&candidate.memory_id)
                && candidate.kind.grounding_role() == GroundingRole::Original
                && candidate
                    .supports_slots
                    .iter()
                    .any(|slot| slot == "evidence_chain_support")
                && candidate.supports_slots.iter().any(|slot| {
                    slot.strip_prefix("evidence_chain_relation:")
                        .is_some_and(|relation| relation_keys.contains(relation))
                })
        }));
    }

    // A normalized profile summary and its modality observations are sibling
    // fields in one typed source record. Carry those descriptive observations
    // into the Claim proof explicitly; they remain body facts and never become
    // native media merely because the record names a modality.
    let profile_summary_memory_ids = selected
        .iter()
        .filter(|atom| {
            atom.subject_binding
                .iter()
                .any(|binding| binding == "structured_profile_summary")
        })
        .map(|atom| atom.memory_id)
        .collect::<HashSet<_>>();
    if !profile_summary_memory_ids.is_empty() {
        selected.extend(atoms.iter().filter(|candidate| {
            profile_summary_memory_ids.contains(&candidate.memory_id)
                && candidate.subject_binding.iter().any(|binding| {
                    binding == "structured_profile_modality_observation"
                })
        }));
    }

    if source_level_qualifier_bridge_allowed(plan) {
        let bridge_atoms = selected
            .iter()
            .copied()
            .filter(|atom| {
                atom.supports_slots.iter().any(|slot| {
                    matches!(
                        slot.as_str(),
                        "source_qualifier_bridge" | "cross_record_qualifier_bridge"
                    )
                })
            })
            .collect::<Vec<_>>();
        let subject = plan.subject.as_deref();
        if !bridge_atoms.is_empty() {
            selected.extend(atoms.iter().filter(|candidate| {
                memory_ids.contains(&candidate.memory_id)
                    && candidate.kind.grounding_role() == GroundingRole::Original
                    && !is_low_information_statement(&candidate.text)
                    && plan
                        .qualifier_terms
                        .iter()
                        .all(|term| text_matches_lexical_term(&candidate.text.to_lowercase(), term))
                    && subject
                        .is_none_or(|person| atom_has_verified_fact_subject(candidate, person))
                    && bridge_atoms.iter().any(|primary| {
                        subject.is_none_or(|person| atom_has_verified_fact_subject(primary, person))
                            && primary.atom_id != candidate.atom_id
                            && !evidence_chain_relation(plan, primary, candidate).is_empty()
                    })
            }));
        }
    }

    // The final answer may paraphrase a fact, but it cannot lose the query's
    // predicate, named entities, or conditions. These are checked against the
    // exact atoms selected for this Claim, after complementary evidence has
    // been added, rather than against concatenated record text.
    if !claim_atoms_cover_query_predicate(plan, &selected)
        || !claim_atoms_cover_named_entities(plan, &selected, &claim_lower)
        || !claim_atoms_cover_conditions(plan, &selected, proof_content_terms)
        // A confident reranker verdict already judged the question's
        // qualifiers ("after recovery from an ankle injury"); otherwise every
        // qualifier term must occur in the selected atoms.
        || !(semantic_coverage(&selected) == Some(true)
            || plan.qualifier_terms.iter().all(|term| {
                selected
                    .iter()
                    .any(|atom| text_matches_lexical_term(&atom.text.to_lowercase(), term))
            }))
    {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    selected.sort_by(|left, right| {
        left.memory_id
            .cmp(&right.memory_id)
            .then_with(|| left.ordinal.cmp(&right.ordinal))
    });
    selected.dedup_by(|left, right| left.atom_id == right.atom_id);
    claim_evidence_binding_fields(&selected)
}

fn claim_answer_value_terms(plan: &QueryPlan, claim_content_terms: &[String]) -> Vec<String> {
    claim_content_terms
        .iter()
        .filter(|term| {
            !plan
                .action_terms
                .iter()
                .any(|action| text_matches_term(term, action) || text_matches_term(action, term))
        })
        .cloned()
        .collect()
}

fn numeric_values_supported(claim: &str, source: &str) -> bool {
    // Match complete signed values with their local unit, never digit substrings.
    static QUANTITY: OnceLock<Regex> = OnceLock::new();
    let re = QUANTITY.get_or_init(|| Regex::new(
        r"(?i)(?:^|[^\p{L}\p{N}_.])([+-]?[0-9]+(?:\.[0-9]+)?)(?:\s*(%|mm|cm|km|kg|mg|mv|kv|ma|ms|usd|eur|volts?|amps?|seconds?|minutes?|hours?|v|a|m|g|s)\b|\s*(%))?"
    ).expect("typed quantity regex"));
    let values = |text: &str| -> Option<Vec<(String, String)>> {
        let mut values = Vec::new();
        for capture in re.captures_iter(text) {
            let value = capture.get(1)?;
            let unit = capture.get(2).or_else(|| capture.get(3));
            let end = unit.map_or(value.end(), |unit| unit.end());
            if text[end..]
                .chars()
                .next()
                .is_some_and(|ch| ch.is_alphanumeric() || ch == '_')
            {
                return None;
            }
            let raw = value.as_str().trim_start_matches(['+', '-']);
            let (integer, fraction) = raw.split_once('.').unwrap_or((raw, ""));
            let integer = integer.trim_start_matches('0');
            let integer = if integer.is_empty() { "0" } else { integer };
            let fraction = fraction.trim_end_matches('0');
            let mut normalized = if fraction.is_empty() {
                integer.to_string()
            } else {
                format!("{integer}.{fraction}")
            };
            if value.as_str().starts_with('-') && normalized != "0" {
                normalized.insert(0, '-');
            }
            let unit = unit
                .map(|unit| match unit.as_str() {
                    "volts" | "volt" | "v" | "V" => "V".to_string(),
                    "amps" | "amp" | "a" | "A" => "A".to_string(),
                    "seconds" | "second" | "s" => "s".to_string(),
                    value => value.to_string(),
                })
                .unwrap_or_default();
            values.push((normalized, unit));
        }
        Some(values)
    };
    let (Some(required), Some(available)) = (values(claim), values(source)) else {
        return false;
    };
    !required.is_empty() && required.iter().all(|value| available.contains(value))
}

/// NLI is only a veto: a bound Claim whose cited source sentences strongly
/// contradict it ("I never liked cooking" vs "James loves cooking") is
/// rejected. Dates and counts are derived by arithmetic over several atoms, so
/// sentence-level entailment does not apply to them.
fn semantic_contradiction_veto(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    atom_ids: &[String],
    claim: &str,
) -> bool {
    if atom_ids.is_empty()
        || matches!(plan.answer_target, AnswerTarget::Temporal | AnswerTarget::Count)
    {
        return false;
    }
    let Some(scorer) = crate::semantic_scorer::active_scorer() else {
        return false;
    };
    let premise = atoms
        .iter()
        .filter(|atom| {
            atom_ids.contains(&atom.atom_id)
                && atom.kind.grounding_role() == GroundingRole::Original
                && matches!(atom.kind, EvidenceKind::Statement | EvidenceKind::Event)
        })
        .map(|atom| match &atom.speaker {
            Some(speaker) => format!("{speaker}: {}", atom.text),
            None => atom.text.clone(),
        })
        .collect::<Vec<_>>()
        .join(" ");
    if premise.is_empty() {
        return false;
    }
    scorer
        .entailment(&[(premise.as_str(), claim)])
        .ok()
        .and_then(|scores| scores.first().copied())
        .is_some_and(|scores| scores.contradiction >= crate::semantic_scorer::CONTRADICTION_VETO)
}

/// Semantic verdict on whether the cited original atoms are about what was
/// asked: the most relevant atom decides when its score is confident, `None`
/// when no scorer ran or the score is in the uncertain middle.
fn semantic_coverage(atoms: &[&EvidenceAtom]) -> Option<bool> {
    let best = atoms
        .iter()
        .filter(|atom| atom.kind.grounding_role() == GroundingRole::Original)
        .filter_map(|atom| atom.semantic_relevance)
        .reduce(f32::max);
    crate::semantic_scorer::relevance_decision(best)
}

fn claim_atoms_cover_query_predicate(plan: &QueryPlan, atoms: &[&EvidenceAtom]) -> bool {
    if plan.action_terms.is_empty() {
        return true;
    }
    if let Some(covered) = semantic_coverage(atoms) {
        return covered;
    }
    // Lexical fallback below: consulted only without a confident semantic
    // score (no scorer configured, or the uncertain band, where general
    // questions are rejected by `semantic_relevance_gate_rejects`).
    // "How did John describe the team bond?": the subject's own statement
    // about the object is the description; the reporting verb never occurs.
    const REPORTING_VERBS: &[&str] = &[
        "describe", "described", "describes", "say", "said", "says", "call", "called",
        "calls", "mention", "mentioned", "mentions", "tell", "told", "tells",
    ];
    if plan
        .action_terms
        .iter()
        .all(|term| REPORTING_VERBS.contains(&term.to_lowercase().as_str()))
        && atoms.iter().any(|atom| {
            atom.kind.grounding_role() == GroundingRole::Original
                && plan
                    .subject
                    .as_deref()
                    .is_some_and(|subject| atom_has_verified_fact_subject(atom, subject))
                && plan
                    .object_terms
                    .iter()
                    .any(|term| text_matches_term(&atom.text.to_lowercase(), term))
        })
    {
        return true;
    }
    atoms.iter().any(|atom| {
        atom.kind.grounding_role() == GroundingRole::Original
            && (plan_requests_structured_profile(plan)
                && atom
                    .subject_binding
                    .iter()
                    .any(|binding| binding == "structured_profile_summary")
                || co_participant_event_bridge(plan, atom)
                || plan.achievement_extremum
                    && atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "achievement_extremum")
                || (atom.supports_slots.iter().any(|slot| slot == "action")
                    && plan
                        .action_terms
                        .iter()
                        .any(|term| text_matches_term(&atom.text.to_lowercase(), term))))
    })
}

fn claim_atoms_cover_named_entities(
    plan: &QueryPlan,
    atoms: &[&EvidenceAtom],
    claim_lower: &str,
) -> bool {
    plan.named_entities
        .iter()
        .filter(|entity| text_matches_named_entity(claim_lower, entity))
        .all(|entity| {
            atoms.iter().any(|atom| {
                atom.kind.grounding_role() == GroundingRole::Original
                    && text_matches_named_entity(&atom.text.to_lowercase(), entity)
            })
        })
}

fn claim_atoms_cover_conditions(
    plan: &QueryPlan,
    atoms: &[&EvidenceAtom],
    claim_content_terms: &[String],
) -> bool {
    plan.condition_terms.is_empty()
        || atoms.iter().any(|atom| {
            let lower = atom.text.to_lowercase();
            strict_condition_terms_match_evidence(plan, &lower)
                && (claim_content_terms.is_empty()
                    || claim_content_terms
                        .iter()
                        .any(|term| text_matches_lexical_term(&lower, term)))
        })
}

fn claim_content_terms(claim_terms: &[String], claim_names: &[String]) -> Vec<String> {
    const SCAFFOLDING_TERMS: &[&str] = &[
        "a",
        "an",
        "and",
        "answer",
        "available",
        "because",
        "by",
        "direct",
        "evidence",
        "fact",
        "for",
        "from",
        "in",
        "is",
        "it",
        "known",
        "likely",
        "not",
        "of",
        "partial",
        "the",
        "this",
        "to",
        "was",
        "with",
    ];
    let claim_names = claim_names.iter().collect::<HashSet<_>>();
    claim_terms
        .iter()
        .filter(|term| !claim_names.contains(term) && !SCAFFOLDING_TERMS.contains(&term.as_str()))
        .cloned()
        .collect()
}

/// A structured-record Claim (`Key: value; Key: value`) quotes several fields
/// of one record. The projector emits one atom per field, so no single atom
/// holds every term. Bind it only when every value exactly equals an original
/// atom of the same cited record; any paraphrase falls through to the normal
/// single-atom verifier.
fn structured_field_claim_bindings(
    atoms: &[EvidenceAtom],
    claim: &str,
    memory_ids: &HashSet<u64>,
) -> Option<(Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>)> {
    let values = claim
        .trim()
        .split("; ")
        .map(|segment| {
            let (key, value) = segment.split_once(": ")?;
            let key_is_label = !key.is_empty()
                && key.len() <= 32
                && key.starts_with(|ch: char| ch.is_ascii_uppercase())
                && key.chars().all(|ch| ch.is_ascii_alphabetic() || ch == ' ');
            let value = normalize_for_dedup(value);
            (key_is_label && !value.is_empty()).then_some(value)
        })
        .collect::<Option<Vec<_>>>()?;
    if values.len() < 2 {
        return None;
    }
    let mut candidate_ids = memory_ids.iter().copied().collect::<Vec<_>>();
    candidate_ids.sort_unstable();
    candidate_ids.into_iter().find_map(|memory_id| {
        let selected = values
            .iter()
            .map(|value| {
                atoms.iter().find(|atom| {
                    atom.memory_id == memory_id
                        && atom.kind.grounding_role() == GroundingRole::Original
                        && normalize_for_dedup(&clean_answer_statement(&atom.text)) == *value
                })
            })
            .collect::<Option<Vec<_>>>()?;
        let mut bindings = claim_evidence_binding_fields(&selected);
        bindings.2.push("structured_record_fields_v1".into());
        Some(bindings)
    })
}

fn claim_evidence_binding_fields(
    selected: &[&EvidenceAtom],
) -> (Vec<String>, Vec<AnswerEvidenceSourceAnchor>, Vec<String>) {
    let evidence_atom_ids = selected
        .iter()
        .map(|atom| atom.atom_id.clone())
        .collect::<Vec<_>>();
    let mut anchor_keys = HashSet::new();
    let evidence_source_anchors = selected
        .iter()
        .filter_map(|atom| atom.source_anchor.clone())
        .filter(|anchor| anchor_keys.insert(format!("{}:{}", anchor.kind, anchor.value)))
        .collect::<Vec<_>>();
    let mut derivation = selected
        .iter()
        .flat_map(|atom| atom.supports_slots.iter())
        .filter_map(|slot| {
            slot.strip_prefix("evidence_chain_relation:")
                .map(str::to_string)
                .or_else(|| {
                    (slot == "evidence_chain_support")
                        .then(|| "complementary_evidence_chain".to_string())
                })
        })
        .collect::<Vec<_>>();
    if selected
        .iter()
        .map(|atom| atom.memory_id)
        .collect::<HashSet<_>>()
        .len()
        > 1
        && derivation.is_empty()
    {
        derivation.push("multiple_source_evidence".to_string());
    }
    derivation.sort();
    derivation.dedup();
    (evidence_atom_ids, evidence_source_anchors, derivation)
}

fn scoped_atom_text(atom: &EvidenceAtom) -> String {
    match atom.kind {
        EvidenceKind::SessionDate | EvidenceKind::SystemMetadata => atom.text.clone(),
        EvidenceKind::NativeMedia => format!("[{}: {}]", atom.modality, atom.text),
        _ => atom
            .speaker
            .as_ref()
            .map(|speaker| format!("{speaker}: {}", atom.text))
            .unwrap_or_else(|| atom.text.clone()),
    }
}

pub(crate) fn finalize_answer_pipeline_receipt(
    mut receipt: AnswerPipelineReceipt,
    coverage: &[CoverageReport],
    draft_mode: &str,
    generator: &str,
    legacy_fallback_used: bool,
    attempted_operators: &[String],
    selected_operator: Option<&str>,
) -> AnswerPipelineReceipt {
    receipt.answer_generator = generator.to_string();
    receipt.legacy_fallback_used = legacy_fallback_used;
    receipt.operator_plan.attempted_operators = attempted_operators.to_vec();
    receipt.operator_plan.selected_operator = selected_operator.map(str::to_string);
    receipt.claim_plan.clear();
    if !coverage.is_empty() {
        receipt.claim_plan = coverage
            .iter()
            .enumerate()
            .map(|(index, report)| AnswerClaimPlanReceipt {
                claim_id: format!("claim-{}", index + 1),
                claim_type: if draft_mode.contains("inferred") {
                    "inference".to_string()
                } else {
                    "fact".to_string()
                },
                text: Some(report.claim.clone()),
                required_slots: receipt.query_plan.required_slots.clone(),
                evidence_ids: report.evidence_ids.clone(),
                evidence_atom_ids: report.evidence_atom_ids.clone(),
                evidence_source_anchors: report.evidence_source_anchors.clone(),
                derivation: report.derivation.clone(),
                support_status: if report.covered && draft_mode.contains("inferred") {
                    "inferred".to_string()
                } else if report.covered {
                    "supported".to_string()
                } else {
                    "unsupported".to_string()
                },
            })
            .collect();
    }

    let unsupported_claim = coverage.iter().any(|report| !report.covered);
    let slots_sufficient = receipt
        .slot_coverage
        .iter()
        .all(|slot| slot.status == "covered");
    receipt.final_evidence_ids = coverage
        .iter()
        .filter(|report| report.covered)
        .flat_map(|report| report.evidence_ids.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();

    if let Some(stage) = receipt
        .stage_trace
        .iter_mut()
        .find(|stage| stage.stage == "claim_plan")
    {
        stage.status = if coverage.is_empty() {
            "failed"
        } else if unsupported_claim {
            "partial"
        } else {
            "completed"
        }
        .to_string();
        stage.output_count = coverage.iter().filter(|report| report.covered).count();
        stage.evidence_ids = receipt.final_evidence_ids.clone();
        stage.reason = unsupported_claim
            .then(|| "one or more generated claims lack supporting original evidence".to_string());
    }
    if let Some(stage) = receipt
        .stage_trace
        .iter_mut()
        .find(|stage| stage.stage == "independent_verification")
    {
        stage.status = if coverage.is_empty() { "failed" } else if unsupported_claim { "partial" } else { "completed" }.to_string();
        stage.output_count = coverage.iter().filter(|report| report.covered).count();
        stage.evidence_ids = receipt.final_evidence_ids.clone();
        stage.reason = unsupported_claim.then(|| "unsupported Claims are withheld from delivery".to_string());
    }
    if let Some(stage) = receipt
        .stage_trace
        .iter_mut()
        .find(|stage| stage.stage == "answer")
    {
        stage.status = if draft_mode == "unavailable" || coverage.is_empty() {
            "failed"
        } else if unsupported_claim || !slots_sufficient {
            "partial"
        } else {
            "completed"
        }
        .to_string();
        stage.output_count = usize::from(draft_mode != "unavailable" && !coverage.is_empty());
        stage.evidence_ids = receipt.final_evidence_ids.clone();
        stage.reason = (draft_mode == "unavailable")
            .then(|| "selected evidence did not produce an answer".to_string());
    }

    if let Some(stage) = receipt
        .stage_trace
        .iter_mut()
        .find(|stage| stage.stage == "verified_delivery")
    {
        stage.status = if draft_mode == "unavailable" || coverage.is_empty() { "failed" } else if unsupported_claim || !slots_sufficient { "partial" } else { "completed" }.to_string();
        stage.output_count = coverage.iter().filter(|report| report.covered).count();
        stage.evidence_ids = receipt.final_evidence_ids.clone();
        stage.reason = (draft_mode == "unavailable").then(|| "no verified Claim was deliverable".to_string());
    }

    receipt.failure_class = if receipt.evidence_plan.candidate_memory_ids.is_empty() {
        "retrieval_miss"
    } else if receipt.evidence_plan.selected_memory_ids.is_empty() {
        "slot_rerank_drop"
    } else if coverage.is_empty() || unsupported_claim {
        "claim_planning_error"
    } else if draft_mode == "unavailable" {
        "answer_expression_error"
    } else if !slots_sufficient {
        "insufficient_evidence"
    } else {
        "none"
    }
    .to_string();
    receipt
}

pub(crate) fn requires_full_space_answer_records(query: &str) -> bool {
    extract_structural_selectors(query)
        .iter()
        .any(|selector| selector.starts_with("chapter:"))
}

fn build_operator_plan(
    plan: &QueryPlan,
    full_space_renderers_allowed: bool,
) -> Vec<AnswerOperator> {
    let mut operators = Vec::with_capacity(29);
    if plan.duration_clauses.is_some() {
        operators.push(AnswerOperator::UnifiedGeneral);
    }
    if full_space_renderers_allowed {
        operators.extend([
            AnswerOperator::NovelChapterRangeFromRecords,
            AnswerOperator::NovelChapterFromRecords,
        ]);
    }
    operators.extend([
        AnswerOperator::OperationalHabit,
        AnswerOperator::VerifiedHandoff,
        AnswerOperator::OperatorBriefing,
        AnswerOperator::AmbientCapture,
        AnswerOperator::LearningResume,
        AnswerOperator::DocumentComparison,
        AnswerOperator::ExperienceRecall,
        AnswerOperator::UniversalProfile,
        AnswerOperator::DesignStyle,
        AnswerOperator::MultimodalAppearanceChange,
        AnswerOperator::RecentVisualCreation,
        AnswerOperator::ResearchObject,
        AnswerOperator::LinkedOrigin,
        AnswerOperator::NovelCrossChapter,
        AnswerOperator::NovelChapter,
        AnswerOperator::NovelChapterUnavailable,
        AnswerOperator::NovelStyle,
    ]);
    let inference_first = plan.inference_type != InferenceType::Direct;
    let temporal_first = matches!(
        plan.answer_target,
        AnswerTarget::Temporal | AnswerTarget::Duration
    );
    if inference_first {
        operators.push(AnswerOperator::GroundedInference);
    }
    if temporal_first {
        operators.extend([AnswerOperator::EventDate, AnswerOperator::TemporalReasoning]);
    }

    // Aspect sets need claim-level entity binding. The legacy list renderer
    // flattens nearby statements before that binding is applied, so the
    // unified answer plan owns them.
    let list_first =
        plan.cardinality == Cardinality::List && plan.answer_shape != AnswerShape::AspectSet;
    let count_first = plan.answer_shape == AnswerShape::CountAggregate;
    let focused_scalar_first = plan.answer_target == AnswerTarget::Duration;
    if list_first {
        operators.push(AnswerOperator::GroundedList);
    }
    if count_first {
        operators.push(AnswerOperator::CrossSourceEventCount);
    }
    if focused_scalar_first {
        operators.push(AnswerOperator::FocusedScalar);
    }
    // Categorical attributes have a closed evidence vocabulary. Run their
    // direct renderer before the general one so a nearby acknowledgement
    // cannot masquerade as an identity or relationship-status answer.
    if plan.categorical_attribute_kind.is_some() {
        operators.push(AnswerOperator::CategoricalAttribute);
    }
    // General questions are owned by the unified generator. Typed list
    // synthesis remains ahead of it because it can report missing list items.
    operators.push(AnswerOperator::UnifiedGeneral);
    if plan.categorical_attribute_kind.is_none() {
        operators.push(AnswerOperator::CategoricalAttribute);
    }
    operators.push(AnswerOperator::SharedLocation);
    if !count_first {
        operators.push(AnswerOperator::CrossSourceEventCount);
    }
    if !focused_scalar_first {
        operators.push(AnswerOperator::FocusedScalar);
    }
    if !list_first {
        operators.push(AnswerOperator::GroundedList);
    }
    if !inference_first {
        operators.push(AnswerOperator::GroundedInference);
    }
    if !temporal_first {
        operators.extend([AnswerOperator::EventDate, AnswerOperator::TemporalReasoning]);
    }
    operators.push(AnswerOperator::AnswerabilityGate);
    operators
}

pub(crate) fn persisted_evidence_atoms_for_memory(
    memory: &MemoryRecord,
) -> Vec<PersistedEvidenceAtom> {
    let source_digest = reasoning_source_fingerprint_for_memory(memory);
    project_source(ProjectionInput {
        memory_id: memory.id,
        content: &memory.content,
        source: &memory.source,
        entity: memory.entity.as_deref(),
        score: 1.0,
        is_event: memory.event_metadata.is_some()
            || memory.content.trim_start().starts_with("event:"),
    })
    .into_iter()
    .map(|atom| {
        let metadata = persisted_evidence_metadata(memory, &atom, &source_digest);
        PersistedEvidenceAtom {
            memory_id: memory.id as i64,
            conversation_id: memory.conversation_id.clone(),
            atom_id: atom.atom_id,
            atom_kind: atom.kind.as_str().to_string(),
            modality: atom.modality,
            ordinal: atom.ordinal as i32,
            speaker: atom.speaker,
            entity: atom.entity,
            content_digest: format!(
                "sha256:{}",
                crate::shared_object::stable_digest(atom.text.as_bytes())
            ),
            body: atom.text,
            source: atom.source,
            source_version: memory.version as i32,
            metadata,
        }
    })
    .collect()
}

/// Keep source identity independent of answer scoring so an old persisted
/// projection can be invalidated even when a record's retrieval rank changes.
pub(crate) fn reasoning_source_fingerprint_for_memory(memory: &MemoryRecord) -> String {
    let input = serde_json::to_vec(&(
        &memory.content,
        &memory.source,
        &memory.entity,
        memory.version,
        memory.event_metadata.is_some() || memory.content.trim_start().starts_with("event:"),
    ))
    .expect("memory source projection input is serializable");
    crate::shared_object::stable_digest(&input)
}

/// This is a source scaffold, not a semantic verdict. Relations and
/// conditions remain empty until the separately verified model proof graph is
/// persisted. Event observations receive immutable source-position identities
/// so later merging never falls back to normalized text or same-day wording.
fn persisted_evidence_metadata(memory: &MemoryRecord, atom: &EvidenceAtom, source_digest: &str) -> serde_json::Value {
    let source_span = serde_json::json!({
        "atom_id": atom.atom_id,
        "start": 0,
        "end": atom.text.len(),
        "text": atom.text,
    });
    let event_observations = if atom.kind == EvidenceKind::Event {
        vec![serde_json::json!({
            "observation_id": format!(
                "source_observation:{}",
                crate::shared_object::stable_digest(
                    &serde_json::to_vec(&(
                        memory.id,
                        memory.version,
                        &source_digest,
                        &atom.atom_id,
                        atom.ordinal,
                    ))
                    .expect("source observation identity is serializable")
                )
            ),
            "source_atom_id": atom.atom_id,
            "multiplicity": null,
            "status": "unverified_source_observation",
            "source_span": source_span,
            "native_event_identity": null,
            "merge_basis": {
                "kind": "immutable_source_position",
                "memory_id": memory.id,
                "source_version": memory.version,
                "source_digest": source_digest,
                "atom_id": atom.atom_id,
                "start": 0,
                "end": atom.text.len(),
                "cross_observation_identity": "unresolved"
            }
        })]
    } else {
        Vec::new()
    };
    serde_json::json!({
        "projection_format": "typed_evidence_atom_v3_reasoning_source",
        "memory_layer": format!("{:?}", memory.layer).to_ascii_lowercase(),
        "source_anchor": atom.source_anchor,
        "section_scope": atom.section_scope,
        "attribution": {
            "fact_subjects": atom.fact_subjects,
            "mentioned_people": atom.mentioned_people,
            "addressed_people": atom.addressed_people,
            "possessive_owners": atom.possessive_owners,
            "subject_binding": atom.subject_binding,
        },
        "reasoning_source_scaffold": {
            "projection_format": "answer_reasoning_source_scaffold_v1",
            "contract_version": crate::answer_reasoning_contract::CONTRACT_VERSION,
            "source_identity": {
                "memory_id": memory.id,
                "source_version": memory.version,
                "source_digest": source_digest,
                "atom_id": atom.atom_id,
                "body_digest": format!("sha256:{}", crate::shared_object::stable_digest(atom.text.as_bytes())),
            },
            "relations": [],
            "conditions": [],
            "event_observations": event_observations,
            "semantic_status": "source_scaffold_unverified",
            "reuse_rule": "never answer from this scaffold; only a separately verified proof graph may authorize a Claim"
        }
    })
}

fn count_question_requested(query_lower: &str) -> bool {
    // "The highest number of points" asks for one extreme value, not for how
    // many events occurred.
    static SUPERLATIVE_NUMBER_OF: OnceLock<Regex> = OnceLock::new();
    if SUPERLATIVE_NUMBER_OF
        .get_or_init(|| {
            Regex::new(r"\b(?:highest|greatest|largest|biggest|most|maximum|max|record|best|top|lowest|smallest|fewest|minimum)\s+number\s+of\b")
                .expect("superlative number-of regex")
        })
        .is_match(query_lower)
        && !query_lower.contains("how many")
    {
        return false;
    }
    contains_any(
        query_lower,
        &["how many", "number of", "多少", "几次", "几个", "数量"],
    ) || word_regex()
        .find_iter(query_lower)
        .any(|value| value.as_str() == "count")
}



fn build_query_plan(query: &str, items: &[RetrievedItem]) -> QueryPlan {
    let lower = query.to_lowercase();
    let categorical_attribute_kind = crate::memory_engine_summary::categorical_answer_kind(query);
    let aspect_set_requested = asks_for_aspect_set(&lower);
    let strategy_set_requested = asks_for_strategy_set(&lower);
    let achievement_extremum = query_requests_achievement_extremum(&lower);
    let media_provenance_requested = query_requests_media_provenance(&lower);
    let recommendation_requested = query_requests_preference_recommendation(&lower);
    let household_dependent_preference = query_requests_household_dependent_preference(&lower);
    let requires_complementary_action =
        strategy_set_requested && query_requires_complementary_action(&lower);
    // A typed list phrase such as "Where has ...?" may not contain a plural
    // noun, so detect it before cardinality. Otherwise the list renderer is
    // unreachable even when the query unambiguously requests multiple values.
    let requested_grounded_list_kind = (!aspect_set_requested && !strategy_set_requested)
        .then(|| crate::memory_engine_summary::grounded_list_kind(query))
        .flatten();
    let typed_list_cardinality_requested = requested_grounded_list_kind
        .is_some_and(|kind| grounded_list_kind_implies_list_cardinality(kind, &lower));
    let cardinality = if count_question_requested(&lower) {
        Cardinality::Count
    } else if household_dependent_preference
        || aspect_set_requested
        || typed_list_cardinality_requested
        || contains_any(
            &lower,
            &[
                "what events",
                "what activities",
                "what projects",
                "what aspects",
                "which aspects",
                "list ",
                "哪些",
                "列出",
                "有什么",
                "做了什么",
                "分别",
            ],
        )
        || contains_whole_ascii_word(&lower, "all")
        || open_plural_list_regex().is_match(&lower)
    {
        Cardinality::List
    } else {
        Cardinality::Single
    };
    let temporal_constraint = extract_temporal_constraint(query);
    let time_terms = extract_time_terms(query);
    let quantity_terms = extract_quantity_terms(query)
        .into_iter()
        .filter(|term| {
            !time_terms.iter().any(|time| {
                time.as_str() == term.as_str()
                    || time
                        .split(|character: char| !character.is_ascii_alphanumeric())
                        .any(|token| {
                            token == term.as_str()
                                || (term.chars().all(|character| character.is_ascii_digit())
                                    && token.strip_prefix(term.as_str()).is_some_and(|suffix| {
                                        matches!(suffix, "st" | "nd" | "rd" | "th")
                                    }))
                        })
            })
        })
        .collect::<Vec<_>>();
    let list_scope = if aspect_set_requested {
        "aspect_set".to_string()
    } else {
        detect_list_scope(&lower, cardinality, &quantity_terms)
    };

    let inference_type = if contains_any(
        &lower,
        &[
            "what if",
            "would have",
            "counterfactual",
            "if they had",
            "如果",
            "假如",
            "反事实",
        ],
    ) {
        InferenceType::Counterfactual
    } else if asks_speculatively(&lower) {
        InferenceType::Likely
    } else {
        InferenceType::Direct
    };
    let temporal_scope = if contains_any(
        &lower,
        &[
            "latest",
            "most recent",
            "recently",
            "最近",
            "最新",
            "上一次",
        ],
    ) {
        "latest"
    } else if contains_any(
        &lower,
        &["between", "from ", "through", "range", "之间", "从", "到"],
    ) {
        "range"
    } else if contains_any(
        &lower,
        &[
            "when",
            "what date",
            "what day",
            "which day",
            "what month",
            "which month",
            "date",
            "before",
            "after",
            "last night",
            "last week",
            "ago",
            "何时",
            "什么时候",
            "哪天",
            "日期",
            "之前",
            "之后",
            "昨晚",
            "上周",
        ],
    ) || temporal_constraint.is_active()
        || requires_relative_time_reference(&time_terms)
        || contains_date_like_value(&lower)
    {
        "bounded"
    } else {
        "any"
    }
    .to_string();
    let temporal_granularity = temporal_constraint
        .granularity()
        .unwrap_or_else(|| detect_temporal_granularity(&lower));
    let requested_modalities = requested_modalities(&lower);
    let people = discover_people(query, items);
    let subject = discover_subject(query, items);
    let subject_is_action_agent = query_subject_is_action_agent(&lower, subject.as_deref());
    let co_participant_event_requested =
        people.len() >= 2 && query_requests_co_participant_event(&lower);
    let named_entities = discover_named_entities(query, &people, subject.as_deref(), &time_terms);
    let answer_target = detect_answer_target(&lower, cardinality);
    let grounded_list_kind = (cardinality == Cardinality::List)
        .then_some(requested_grounded_list_kind)
        .flatten();
    let list_requires_household_context =
        grounded_list_kind.is_some() && query_requests_household_list_context(&lower);
    let structural_selectors = extract_structural_selectors(query);
    let mut answer_shape = detect_answer_shape(
        answer_target,
        cardinality,
        &list_scope,
        inference_type,
        aspect_set_requested,
        strategy_set_requested,
    );
    if answer_target == AnswerTarget::Event
        && structural_selectors.len() > 1
        && answer_shape == AnswerShape::Scalar
    {
        answer_shape = AnswerShape::ActionSet;
    }
    let mut query_terms = question_informative_terms(query);
    let temporal_control_terms = temporal_constraint.control_terms();
    query_terms.retain(|term| !temporal_control_terms.iter().any(|control| term == control));
    let negation_terms = extract_negation_terms(&lower);
    let recommendation_topic_terms = recommendation_requested
        .then(|| {
            recommendation_topic_terms(&query_terms, &people, subject.as_deref(), &named_entities)
        })
        .unwrap_or_default();
    let condition_relation = extract_condition_relation(query);
    let condition_terms = extract_condition_terms(query);
    let mut action_terms = query_action_terms(
        &query_terms,
        &people,
        &named_entities,
        &time_terms,
        &quantity_terms,
        &negation_terms,
    );
    // The ordinary predicate classifier remains authoritative when it found
    // a verb. The local auxiliary/subject form is a recovery path for verbs
    // outside that conservative vocabulary (for example, "paint" or "move").
    // Adding both makes conjunctions and prepositions look like extra actions.
    if action_terms.is_empty() {
        action_terms.extend(subject_local_action_terms(&lower, subject.as_deref()));
    }
    action_terms = dedup_terms(action_terms);
    let participant_relationship_qualifier = participant_relationship_qualifier_query_terms(query);
    let qualifier_query_terms = participant_relationship_qualifier
        .as_ref()
        .cloned()
        .unwrap_or_else(|| relational_qualifier_query_terms(query));
    let mut qualifier_action_terms = query_action_terms(
        &qualifier_query_terms,
        &people,
        &named_entities,
        &time_terms,
        &quantity_terms,
        &negation_terms,
    );
    qualifier_action_terms.extend(
        qualifier_query_terms
            .iter()
            .filter(|term| relationship_communication_term(term))
            .cloned(),
    );
    let qualifier_action_terms = dedup_terms(qualifier_action_terms);
    action_terms.retain(|term| {
        !qualifier_action_terms
            .iter()
            .any(|qualifier| qualifier == term)
    });
    let qualifier_terms = query_object_terms(
        &qualifier_query_terms,
        &people,
        subject.as_deref(),
        &named_entities,
        &time_terms,
        &quantity_terms,
        &negation_terms,
        &condition_terms,
        &[],
    );
    let mut object_terms = query_object_terms(
        &query_terms,
        &people,
        subject.as_deref(),
        &named_entities,
        &time_terms,
        &quantity_terms,
        &negation_terms,
        &condition_terms,
        &action_terms,
    );
    object_terms.retain(|term| {
        !qualifier_terms.iter().any(|qualifier| qualifier == term)
            && !temporal_control_terms.iter().any(|control| term == control)
    });
    if query_has_collective_participant_reference(&lower) {
        // Family identifies the participant group here, not the answer object.
        object_terms.retain(|term| !matches!(term.as_str(), "family" | "families"));
    }
    if grounded_list_kind.is_some() {
        // Request verbs identify a list operation, not literal source terms.
        object_terms.retain(|term| !grounded_list_request_predicate_term(term));
    }
    let answer_class_terms = extract_answer_class_terms(&lower, cardinality, answer_target);
    if recommendation_requested {
        action_terms.retain(|term| !is_recommendation_control_term(term));
        object_terms.retain(|term| !is_recommendation_control_term(term));
    }
    let mut protected_terms = protected_query_terms(
        &people,
        subject.as_deref(),
        &named_entities,
        &time_terms,
        &quantity_terms,
        &negation_terms,
        &condition_terms,
        &object_terms,
        &qualifier_terms,
    );
    protected_terms.extend(answer_class_terms.iter().cloned());
    protected_terms = dedup_terms(protected_terms);
    protected_terms.extend(recommendation_topic_terms.iter().cloned());
    let mut required_slots = vec!["answer_bearing_fact".to_string()];
    if categorical_attribute_kind.is_some() {
        required_slots.push("categorical_attribute_value".to_string());
    }
    if subject.is_some() {
        required_slots.push("subject".to_string());
    }
    for person in &people {
        required_slots.push(person_slot_name(person));
    }
    if !action_terms.is_empty() {
        required_slots.push("action".to_string());
    }
    if requires_action_object_complement(
        answer_target,
        inference_type,
        &action_terms,
        &object_terms,
    ) {
        required_slots.push("action_object_complement".to_string());
    }
    if requires_observed_action_fact(answer_target, subject_is_action_agent, &action_terms) {
        required_slots.push("observed_action_fact".to_string());
    }
    if requires_complementary_action {
        required_slots.push("complementary_action".to_string());
    }
    if achievement_extremum {
        required_slots.push("achievement_extremum".to_string());
    }
    if !object_terms.is_empty() {
        required_slots.push("object".to_string());
    }
    if !structural_selectors.is_empty() {
        if structural_selectors.len() == 1 {
            required_slots.push("structural_selector".to_string());
        }
        for selector in &structural_selectors {
            required_slots.push(structural_selector_slot_name(selector));
        }
    }
    if !answer_class_terms.is_empty() {
        required_slots.push("answer_class".to_string());
    }
    if !qualifier_terms.is_empty() {
        required_slots.push("qualifier_context".to_string());
    }
    if !named_entities.is_empty() {
        required_slots.push("named_entity".to_string());
        required_slots.extend(
            named_entities
                .iter()
                .map(|entity| named_entity_slot_name(entity)),
        );
    }
    if temporal_scope != "any" {
        required_slots.push("temporal_scope".to_string());
    }
    if temporal_constraint.is_active() {
        required_slots.push("temporal_constraint".to_string());
    }
    if requires_relative_time_reference(&time_terms) {
        required_slots.push("relative_time_reference".to_string());
    }
    if answer_target == AnswerTarget::Duration {
        required_slots.push("duration_value".to_string());
    }
    if answer_target == AnswerTarget::Temporal {
        required_slots.push("event_time".to_string());
    }
    if answer_target == AnswerTarget::Initiation {
        required_slots.push("initiation_event".to_string());
    }
    if answer_target == AnswerTarget::Location {
        required_slots.push("location".to_string());
    }
    if cardinality == Cardinality::Count
        || (cardinality == Cardinality::List
            && quantity_terms
                .iter()
                .any(|term| explicit_quantity_value(term).is_some()))
    {
        required_slots.push("quantity".to_string());
    }
    if cardinality == Cardinality::List {
        required_slots.push("list_scope".to_string());
    }
    if grounded_list_kind.is_some() {
        required_slots.push("list_value".to_string());
        if list_requires_household_context {
            required_slots.push("list_participant_scope".to_string());
        }
    }
    if household_dependent_preference {
        required_slots.push("household_dependent_preference".to_string());
    }
    if inference_type != InferenceType::Direct {
        required_slots.push("inference_basis".to_string());
    }
    if recommendation_requested {
        required_slots.push("recommendation_capability".to_string());
        required_slots.push("recommendation_topic".to_string());
    }
    if !negation_terms.is_empty() {
        required_slots.push("negation".to_string());
    }
    if !condition_terms.is_empty() {
        required_slots.push("condition".to_string());
    }
    if requested_modalities.iter().any(|value| value != "text") {
        required_slots.push("requested_modality".to_string());
    }

    QueryPlan {
        subject,
        subject_is_action_agent,
        people,
        named_entities,
        answer_target,
        answer_shape,
        categorical_attribute_kind,
        grounded_list_kind,
        list_requires_household_context,
        household_dependent_preference,
        requires_complementary_action,
        achievement_extremum,
        temporal_scope,
        media_provenance_requested,
        co_participant_event_requested,
        recommendation_requested,
        recommendation_topic_terms,
        temporal_granularity,
        temporal_constraint,
        duration_clauses: crate::memory_engine_summary::temporal_duration_requested(query)
            .then(|| crate::memory_engine_query::temporal_relation_clauses(query))
            .flatten(),
        time_terms,
        structural_selectors,
        action_terms,
        object_terms,
        answer_class_terms,
        qualifier_terms,
        participant_relationship_qualifier: participant_relationship_qualifier.is_some(),
        cardinality,
        quantity_terms,
        list_scope,
        inference_type,
        negation_terms,
        condition_terms,
        condition_relation,
        protected_terms,
        requested_modalities,
        required_slots,
        query_terms,
        possessive_relation_actors: possessive_relation_actors(&lower),
    }
}

/// "Jolene's parents gave ..." -> ["parents"]. Coordinated phrasing such as
/// "Jolene and her partner" is not a possessed actor and is not returned.
fn possessive_relation_actors(query_lower: &str) -> Vec<String> {
    static POSSESSIVE_RELATION: OnceLock<Regex> = OnceLock::new();
    POSSESSIVE_RELATION
        .get_or_init(|| {
            Regex::new(r"\b[a-z]+(?:'|’)s\s+(parents?|mom|mother|dad|father|brother|sister|son|daughter|grandma|grandmother|grandpa|grandfather|aunt|uncle|cousin|wife|husband|partner|boyfriend|girlfriend|friends?|boss|coach|teacher|neighbou?r|roommate|colleague|teammates?)\b")
                .expect("possessive relation actor regex")
        })
        .captures_iter(query_lower)
        .filter_map(|captures| captures.get(1).map(|value| value.as_str().to_string()))
        .collect()
}

fn query_plan_receipt(plan: &QueryPlan) -> AnswerQueryPlanReceipt {
    AnswerQueryPlanReceipt {
        plan_format: "typed_question_plan_v19".to_string(),
        subject: plan.subject.clone(),
        people: plan.people.clone(),
        named_entities: plan.named_entities.clone(),
        answer_target: plan.answer_target.as_str().to_string(),
        answer_shape: plan.answer_shape.as_str().to_string(),
        requires_complementary_action: plan.requires_complementary_action,
        achievement_extremum: plan.achievement_extremum,
        temporal_scope: plan.temporal_scope.clone(),
        media_provenance_requested: plan.media_provenance_requested,
        co_participant_event_requested: plan.co_participant_event_requested,
        recommendation_requested: plan.recommendation_requested,
        household_dependent_preference: plan.household_dependent_preference,
        recommendation_topic_terms: plan.recommendation_topic_terms.clone(),
        temporal_granularity: plan.temporal_granularity.as_str().to_string(),
        temporal_constraint: plan.temporal_constraint.as_str(),
        time_terms: plan.time_terms.clone(),
        action_terms: plan.action_terms.clone(),
        structural_selectors: plan.structural_selectors.clone(),
        object_terms: plan.object_terms.clone(),
        answer_class_terms: plan.answer_class_terms.clone(),
        qualifier_terms: plan.qualifier_terms.clone(),
        participant_relationship_qualifier: plan.participant_relationship_qualifier,
        cardinality: plan.cardinality.as_str().to_string(),
        quantity_terms: plan.quantity_terms.clone(),
        list_scope: plan.list_scope.clone(),
        inference_type: plan.inference_type.as_str().to_string(),
        negation_terms: plan.negation_terms.clone(),
        condition_terms: plan.condition_terms.clone(),
        protected_terms: plan.protected_terms.clone(),
        requested_modalities: plan.requested_modalities.clone(),
        required_slots: plan.required_slots.clone(),
    }
}

fn evidence_atom_receipt(atom: &EvidenceAtom) -> AnswerEvidenceAtomReceipt {
    AnswerEvidenceAtomReceipt {
        atom_id: atom.atom_id.clone(),
        memory_id: atom.memory_id,
        evidence_type: atom.kind.as_str().into(),
        grounding_role: atom.kind.grounding_role().as_str().into(),
        modality: atom.modality.clone(),
        source: atom.source.clone(),
        source_anchor: atom.source_anchor.clone(),
        speaker: atom.speaker.clone(),
        fact_subjects: atom.fact_subjects.clone(),
        addressed_people: atom.addressed_people.clone(),
        possessive_owners: atom.possessive_owners.clone(),
        subject_binding: atom.subject_binding.clone(),
        score: atom.score,
        supports_slots: atom.supports_slots.clone(),
        semantic_relevance: atom.semantic_relevance,
    }
}

#[derive(Debug, Default)]
struct StructuredProfileProjection {
    scope_subjects: Vec<(String, String)>,
    statements: Vec<String>,
    modality_observations: Vec<(String, String)>,
}

fn normalize_profile_observation_modality(value: &str) -> Option<String> {
    let value = value.trim().to_ascii_lowercase();
    if value.is_empty() {
        return None;
    }
    Some(
        match value.as_str() {
            "photo" | "picture" | "screenshot" | "ui" | "web" | "canvas" => "image",
            "footage" | "animation" | "motion" => "video",
            "voice" | "speech" => "audio",
            "telemetry" | "lidar" | "imu" => "sensor",
            other => other,
        }
        .to_string(),
    )
}

fn structured_profile_projection(content: &str) -> Option<StructuredProfileProjection> {
    let payload = serde_json::from_str::<serde_json::Value>(content).ok()?;
    let profile_type = payload.get("memory_type")?.as_str()?;
    if !matches!(
        profile_type,
        "writing_style_profile"
            | "novel_style_profile"
            | "design_style_profile"
            | "universal_profile"
            | "operational_habit_profile"
    ) {
        return None;
    }

    let mut projection = StructuredProfileProjection::default();
    let mut seen_subjects = HashSet::new();
    for key in ["source_id", "mission_id", "subject", "title", "operator"] {
        let Some(value) = payload
            .get(key)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty() && value.chars().count() <= 256)
        else {
            continue;
        };
        let normalized = normalize_person_key(value);
        if !normalized.is_empty() && seen_subjects.insert(normalized) {
            projection
                .scope_subjects
                .push((key.to_string(), value.to_string()));
        }
    }
    if let Some(summary) = payload
        .get("summary")
        .and_then(serde_json::Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        projection.statements.push(summary.to_string());
    }

    let mut seen_observations = HashSet::new();
    if let Some(records) = payload
        .get("modality_signal_records")
        .and_then(serde_json::Value::as_array)
    {
        for record in records.iter().take(32) {
            let Some(modality) = record
                .get("modality")
                .and_then(serde_json::Value::as_str)
                .and_then(normalize_profile_observation_modality)
            else {
                continue;
            };
            let source = record
                .get("source")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("unspecified source");
            let observation = record
                .get("observation")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .unwrap_or("The profile record declares this evidence modality.");
            let signal_kinds = record
                .get("signal_kinds")
                .and_then(serde_json::Value::as_array)
                .map(|values| {
                    values
                        .iter()
                        .filter_map(serde_json::Value::as_str)
                        .take(8)
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .filter(|value| !value.is_empty());
            let text = match signal_kinds {
                Some(signal_kinds) => format!(
                    "Described {modality} evidence from {source}: {observation} Signals: {signal_kinds}."
                ),
                None => format!("Described {modality} evidence from {source}: {observation}"),
            };
            let key = format!("{modality}:{}", normalize_for_dedup(&text));
            if seen_observations.insert(key) {
                projection.modality_observations.push((modality, text));
            }
        }
    }
    if projection.modality_observations.is_empty() {
        if let Some(modalities) = payload
            .get("evidence_modalities")
            .and_then(serde_json::Value::as_array)
        {
            for modality in modalities
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter_map(normalize_profile_observation_modality)
                .take(16)
            {
                let text = format!(
                    "The profile record declares described {modality} evidence; no native media object is attached to this atom."
                );
                if seen_observations.insert(modality.clone()) {
                    projection.modality_observations.push((modality, text));
                }
            }
        }
    }
    Some(projection)
}

fn project_structured_profile_source(
    input: &ProjectionInput<'_>,
    profile: StructuredProfileProjection,
) -> Vec<EvidenceAtom> {
    let mut atoms = Vec::new();
    let mut ordinal = 0usize;
    for statement in &profile.statements {
        push_atom(
            &mut atoms,
            input,
            &mut ordinal,
            0,
            EvidenceKind::Statement,
            "text",
            statement,
            None,
            None,
        );
        if let Some(atom) = atoms.last_mut() {
            push_distinct_binding(
                &mut atom.subject_binding,
                "structured_profile_summary".to_string(),
            );
        }
    }
    for (modality, observation) in &profile.modality_observations {
        push_atom(
            &mut atoms,
            input,
            &mut ordinal,
            0,
            EvidenceKind::Statement,
            modality,
            observation,
            None,
            None,
        );
        if let Some(atom) = atoms.last_mut() {
            push_distinct_binding(
                &mut atom.subject_binding,
                "structured_profile_modality_observation".to_string(),
            );
        }
    }
    for atom in &mut atoms {
        for (field, subject) in &profile.scope_subjects {
            push_distinct_person(&mut atom.fact_subjects, subject.clone());
            push_distinct_binding(
                &mut atom.subject_binding,
                format!(
                    "structured_profile_scope:{field}:{}",
                    normalize_person_key(subject)
                ),
            );
        }
    }
    atoms
}

thread_local! {
    /// Person names attested by the record being projected; see
    /// `attested_person_names`.
    static ATTESTED_NAMES: std::cell::RefCell<Option<HashSet<String>>> =
        const { std::cell::RefCell::new(None) };
}

fn is_sentence_initial(text: &str, start: usize) -> bool {
    text[..start]
        .trim_end()
        .chars()
        .last()
        .is_none_or(|ch| matches!(ch, '.' | '!' | '?' | ':' | ';' | '"' | '\u{201c}' | '(' | '-' | '\u{2013}' | '\u{2014}'))
}

/// Names a record itself attests: dialogue speakers, the record entity, and
/// capitalized words used mid-sentence. A capitalized word that only ever
/// opens sentences ("Keep", "Got", "Family") is an ordinary word, not a
/// person, and must not take the speaker's facts.
fn attested_person_names(content: &str, entity: Option<&str>) -> HashSet<String> {
    static SPEAKER: OnceLock<Regex> = OnceLock::new();
    static COORDINATED_PERSON: OnceLock<Regex> = OnceLock::new();
    let speaker = SPEAKER.get_or_init(|| {
        Regex::new(r"^(?:[A-Za-z]{0,3}\d+:\d+\s+)?([A-Z][\w'’-]*(?:\s[A-Z][\w'’-]*)?)\s*:\s")
            .expect("speaker label regex")
    });
    let mut names = HashSet::new();
    if let Some(entity) = entity {
        names.insert(normalize_person_key(entity));
    }
    for line in content.lines().map(str::trim) {
        let body = match speaker.captures(line) {
            Some(captures) => {
                let label = captures.get(1).map(|m| m.as_str()).unwrap_or_default();
                names.insert(normalize_person_key(label));
                for part in label.split_whitespace() {
                    names.insert(normalize_person_key(part));
                }
                &line[captures.get(0).map(|m| m.end()).unwrap_or(0)..]
            }
            None => line,
        };
        for capture in person_name_regex().find_iter(body) {
            let raw = capture.as_str();
            // At a sentence start only person-shaped uses attest a name:
            // a possessive ("Jolene's parents") or coordination with a
            // person ("Anthony and I", "Anthony and Jolene").
            let following = &body[capture.end()..];
            let person_shaped = raw.ends_with("'s")
                || raw.ends_with("’s")
                || COORDINATED_PERSON
                    .get_or_init(|| {
                        Regex::new(r"^\s*(?:,\s*[A-Z][a-z]+\s*)?(?:,\s*)?and\s+(?:I|me|my|[A-Z][a-z]+)\b")
                            .expect("coordinated person regex")
                    })
                    .is_match(following);
            if !is_sentence_initial(body, capture.start()) || person_shaped {
                names.insert(normalize_person_key(&strip_person_possessive(raw)));
            }
        }
    }
    names
}

fn project_source(input: ProjectionInput<'_>) -> Vec<EvidenceAtom> {
    struct Restore(Option<HashSet<String>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ATTESTED_NAMES.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let previous = ATTESTED_NAMES
        .with(|slot| slot.replace(Some(attested_person_names(input.content, input.entity))));
    let _restore = Restore(previous);
    project_source_inner(input)
}

fn project_source_inner(input: ProjectionInput<'_>) -> Vec<EvidenceAtom> {
    if let Some(profile) = structured_profile_projection(input.content) {
        return project_structured_profile_source(&input, profile);
    }
    let mut atoms = Vec::new();
    let mut ordinal = 0usize;
    let mut section_scope = Vec::new();
    let mut section_scopes = HashMap::new();
    for (context_group, line) in input
        .content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .enumerate()
    {
        let heading_scope = declared_section_scope(line);
        if crate::answer_execution::answer_worker_cancelled() { break; }
        let inline_scoped_body = line.split_once(':').or_else(|| line.split_once('：'))
            .is_some_and(|(prefix, body)| !body.trim().is_empty()
                && !extract_structural_selectors(prefix).is_empty());
        let structural_heading = heading_scope.as_ref().is_some_and(|scope| !scope.is_empty())
            && !inline_scoped_body;
        if let Some(scope) = heading_scope {
            section_scope = scope;
        }
        section_scopes.insert(context_group, section_scope.clone());
        let lower = line.to_lowercase();
        let source_anchor = dialogue_turn_source_anchor(line);
        if structural_heading {
            // A structural title constrains the following body; it is neither
            // an utterance by a speaker named Section nor an answer fact.
            push_atom(&mut atoms, &input, &mut ordinal, context_group,
                EvidenceKind::SystemMetadata, "metadata", line, None, source_anchor.as_ref());
            continue;
        }
        if is_session_date_line(&lower) {
            push_atom(
                &mut atoms,
                &input,
                &mut ordinal,
                context_group,
                EvidenceKind::SessionDate,
                "time",
                line,
                None,
                source_anchor.as_ref(),
            );
            continue;
        }
        if is_system_metadata_line(&lower) {
            push_atom(
                &mut atoms,
                &input,
                &mut ordinal,
                context_group,
                EvidenceKind::SystemMetadata,
                "metadata",
                line,
                None,
                source_anchor.as_ref(),
            );
            continue;
        }

        let without_media = media_caption_regex().replace_all(line, "");
        let (speaker, body) = split_speaker_line(without_media.trim());
        for capture in media_caption_regex().captures_iter(line) {
            let Some(body) = capture.get(2).map(|value| value.as_str().trim()) else {
                continue;
            };
            let modality = capture
                .get(1)
                .map(|value| value.as_str().to_ascii_lowercase())
                .unwrap_or_else(|| "image".to_string());
            push_atom(
                &mut atoms,
                &input,
                &mut ordinal,
                context_group,
                EvidenceKind::NativeMedia,
                &modality,
                body,
                speaker.as_deref(),
                source_anchor.as_ref(),
            );
        }

        for statement in split_statements(body) {
            if statement.chars().count() < 3 {
                continue;
            }
            push_atom(
                &mut atoms,
                &input,
                &mut ordinal,
                context_group,
                if input.is_event {
                    EvidenceKind::Event
                } else {
                    EvidenceKind::Statement
                },
                "text",
                statement,
                speaker.as_deref(),
                source_anchor.as_ref(),
            );
        }
    }

    if let Some(entity) = input
        .entity
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        push_atom(
            &mut atoms,
            &input,
            &mut ordinal,
            usize::MAX,
            EvidenceKind::Entity,
            "metadata",
            entity,
            None,
            None,
        );
    }
    for atom in &mut atoms {
        atom.section_scope = section_scopes
            .get(&atom.context_group)
            .cloned()
            .unwrap_or_default();
    }
    atoms
}

fn declared_section_scope(line: &str) -> Option<Vec<String>> {
    let mut heading = line.trim().trim_start_matches('#').trim();
    let lower = heading.to_ascii_lowercase();
    for key in [
        "section:", "chapter:", "article:", "page:", "step:", "part:", "episode:", "scene:",
        "act:", "clause:",
    ] {
        if lower.starts_with(key) {
            heading = heading[key.len()..].trim();
            if !heading.is_empty() && heading.chars().all(|ch| ch.is_ascii_digit()) {
                return Some(extract_structural_selectors(&format!(
                    "{} {heading}",
                    key.trim_end_matches(':')
                )));
            }
            return Some(extract_structural_selectors(heading));
        }
    }
    static HEADING: OnceLock<Regex> = OnceLock::new();
    HEADING.get_or_init(|| Regex::new(r"(?i)^(?:(?:chapter|section|article|page|step|part|episode|scene|act|clause)\s+[0-9]+|第\s*[0-9]+\s*[章节条页步部集幕回篇段项款])\s*(?:[:：].*)?$").expect("explicit structural heading"))
        .is_match(heading).then(|| extract_structural_selectors(heading))
}

fn push_atom(
    atoms: &mut Vec<EvidenceAtom>,
    input: &ProjectionInput<'_>,
    ordinal: &mut usize,
    context_group: usize,
    kind: EvidenceKind,
    modality: &str,
    text: &str,
    speaker: Option<&str>,
    source_anchor: Option<&AnswerEvidenceSourceAnchor>,
) {
    let text = text.trim();
    if text.is_empty() {
        return;
    }
    let attribution = fact_attribution(text, speaker);
    let atom_id = format!("memory:{}:atom:{}", input.memory_id, *ordinal);
    atoms.push(EvidenceAtom {
        atom_id,
        memory_id: input.memory_id,
        kind,
        modality: modality.to_string(),
        text: text.to_string(),
        source: input.source.to_string(),
        source_anchor: source_anchor.cloned(),
        speaker: speaker.map(str::to_string),
        entity: input.entity.map(str::to_string),
        section_scope: Vec::new(),
        fact_subjects: attribution.fact_subjects,
        mentioned_people: attribution.mentioned_people,
        addressed_people: attribution.addressed_people,
        possessive_owners: attribution.possessive_owners,
        subject_binding: attribution.subject_binding,
        base_score: input.score,
        score: input.score,
        supports_slots: Vec::new(),
        context_group,
        ordinal: *ordinal,
        semantic_relevance: None,
    });
    *ordinal += 1;
}

fn fact_attribution(statement: &str, speaker: Option<&str>) -> FactAttribution {
    let attested = ATTESTED_NAMES.with(|slot| slot.borrow().clone());
    fact_attribution_with_names(statement, speaker, attested.as_ref())
}

/// `attested`: names the surrounding record attests; when given, a
/// sentence-initial capitalized word outside it is not treated as a person.
fn fact_attribution_with_names(
    statement: &str,
    speaker: Option<&str>,
    attested: Option<&HashSet<String>>,
) -> FactAttribution {
    let mut attribution = FactAttribution::default();
    for capture in person_name_regex().find_iter(statement) {
        let raw = capture.as_str();
        let name = strip_person_possessive(raw);
        if !likely_person_name(&name)
            || sentence_initial_gerund(statement, capture.start(), capture.end())
            || (is_sentence_initial(statement, capture.start())
                && attested.is_some_and(|names| !names.contains(&normalize_person_key(&name))))
        {
            continue;
        }
        if person_occurrence_is_direct_address(statement, capture.start(), capture.end()) {
            push_distinct_person(&mut attribution.addressed_people, name);
            continue;
        }
        push_distinct_person(&mut attribution.mentioned_people, name.clone());
        if raw.ends_with("'s") || raw.ends_with('’') || raw.ends_with("’s") {
            push_distinct_person(&mut attribution.possessive_owners, name.clone());
        }
        if statement_explicitly_attributes_person(statement, &name) {
            push_distinct_person(&mut attribution.fact_subjects, name.clone());
            push_distinct_binding(
                &mut attribution.subject_binding,
                format!("explicit_named_subject:{}", normalize_person_key(&name)),
            );
        }
    }

    if let Some(speaker) = speaker.filter(|value| !value.trim().is_empty()) {
        let speaker_is_already_explicit = attribution
            .fact_subjects
            .iter()
            .any(|subject| normalize_person_key(subject) == normalize_person_key(speaker));
        let explicit_other_subject = attribution
            .fact_subjects
            .iter()
            .any(|subject| normalize_person_key(subject) != normalize_person_key(speaker));
        if speaker_is_already_explicit {
            push_distinct_binding(
                &mut attribution.subject_binding,
                "speaker_matches_explicit_subject".to_string(),
            );
        } else if statement_binds_speaker_as_fact_subject(statement, &attribution.addressed_people)
            && (!explicit_other_subject || first_person_is_fact_subject(statement))
        {
            push_distinct_person(&mut attribution.fact_subjects, speaker.to_string());
            let reason = if statement_has_first_person_reference(statement) {
                "speaker_first_person"
            } else {
                "speaker_elliptical"
            };
            push_distinct_binding(
                &mut attribution.subject_binding,
                format!("{reason}:{}", normalize_person_key(speaker)),
            );
        }
    }
    attribution
}

fn strip_person_possessive(value: &str) -> String {
    value
        .trim()
        .strip_suffix("'s")
        .or_else(|| value.trim().strip_suffix("’s"))
        .unwrap_or(value.trim())
        .to_string()
}

/// "Organizing a camp ..." / "Running the club ...": a capitalized, sentence-
/// initial -ing word followed by a determiner or preposition opens a gerund
/// phrase, not a name. Names such as "Ming visited ..." are unaffected.
fn sentence_initial_gerund(statement: &str, start: usize, end: usize) -> bool {
    let word = &statement[start..end];
    let sentence_initial = statement[..start]
        .trim_end()
        .chars()
        .last()
        .is_none_or(|ch| matches!(ch, '.' | '!' | '?' | ':' | ';' | '"' | '\u{201c}'));
    let next_word = statement[end..]
        .split_whitespace()
        .next()
        .unwrap_or("")
        .trim_matches(|ch: char| !ch.is_alphanumeric())
        .to_lowercase();
    sentence_initial
        && word.len() > 4
        && word.to_lowercase().ends_with("ing")
        && matches!(
            next_word.as_str(),
            "a" | "an" | "the" | "my" | "his" | "her" | "our" | "their" | "your" | "some"
                | "this" | "that" | "these" | "those" | "to" | "with" | "for" | "in" | "on"
                | "at" | "up" | "out" | "around" | "into" | "onto" | "over" | "through"
                | "back" | "off" | "down"
        )
}

/// Capitalized words that open sentences but are never names ("Some close
/// friends kept supporting me" must not create a person called "Some").
const CAPITALIZED_NON_NAMES: &[&str] = &[
    "some", "many", "most", "all", "every", "each", "any", "both", "few", "several", "none",
    "no", "not", "these", "those", "that", "there", "here", "what", "when", "where", "why",
    "how", "which", "who", "whose", "yes", "yeah", "yep", "yup", "nope", "hey", "hi", "hello",
    "thanks", "thank", "sorry", "well", "oh", "wow", "ok", "okay", "so", "just", "also",
    "maybe", "perhaps", "today", "tonight", "yesterday", "tomorrow", "last", "next", "now",
    "then", "after", "before", "since", "while", "if", "because", "although", "though",
    "their", "its", "your", "everyone", "someone", "anyone", "nobody", "everybody",
    "somebody", "something", "everything", "nothing", "anything", "definitely", "absolutely",
    "totally", "really", "honestly", "actually", "anyway", "plus", "still", "even", "only",
    "great", "awesome", "cool", "nice", "good", "glad", "congrats", "congratulations", "haha",
    "lol", "omg", "aww", "ah", "um", "uh", "hmm", "sure", "right", "exactly", "agreed", "let",
];

fn likely_person_name(value: &str) -> bool {
    let key = normalize_person_key(value);
    !key.is_empty()
        && !CAPITALIZED_NON_NAMES.contains(&key.as_str())
        && !matches!(
            key.as_str(),
            "a" | "an"
                | "and"
                | "but"
                | "for"
                | "from"
                | "he"
                | "her"
                | "his"
                | "i"
                | "in"
                | "it"
                | "my"
                | "our"
                | "she"
                | "section"
                | "chapter"
                | "thanks"
                | "thank"
                | "the"
                | "they"
                | "this"
                | "we"
                | "you"
        )
}

fn push_distinct_person(values: &mut Vec<String>, value: String) {
    if !values
        .iter()
        .any(|existing| normalize_person_key(existing) == normalize_person_key(&value))
    {
        values.push(value);
    }
}

fn push_distinct_binding(values: &mut Vec<String>, value: String) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn person_occurrence_is_direct_address(statement: &str, start: usize, end: usize) -> bool {
    let before = statement[..start].trim_end().to_ascii_lowercase();
    let after = statement[end..].trim_start();
    if after.starts_with(',') && before.is_empty() {
        return true;
    }
    let greeting = before
        .trim_end_matches(|character: char| matches!(character, ',' | ':' | '-' | '—'))
        .split_whitespace()
        .next_back()
        .unwrap_or_default();
    matches!(
        greeting,
        "thanks" | "thank" | "hi" | "hello" | "hey" | "dear"
    ) || (before.is_empty() && after.starts_with(','))
}

fn statement_explicitly_attributes_person(statement: &str, person: &str) -> bool {
    let person = person.trim();
    if person.is_empty() {
        return false;
    }
    let lower = statement.to_ascii_lowercase();
    let person_lower = person.to_ascii_lowercase();
    lower.match_indices(&person_lower).any(|(start, _)| {
        let end = start + person_lower.len();
        let before_character = lower[..start].chars().next_back();
        let after_character = lower[end..].chars().next();
        if before_character.is_some_and(|value| value.is_ascii_alphanumeric())
            || after_character.is_some_and(|value| value.is_ascii_alphanumeric())
            || person_occurrence_is_direct_address(statement, start, end)
        {
            return false;
        }
        let before = lower[..start].trim_end();
        let after = lower[end..].trim_start();
        if after.starts_with("'s") || after.starts_with("’s") {
            // A leading owned attribute describes its owner, unlike an owner
            // mentioned inside another actor's predicate.
            return before.is_empty();
        }
        let previous = before
            .split(|character: char| !character.is_ascii_alphabetic())
            .filter(|value| !value.is_empty())
            .next_back()
            .unwrap_or_default();
        if matches!(
            previous,
            "about"
                | "asked"
                | "at"
                | "for"
                | "from"
                | "met"
                | "of"
                | "saw"
                | "said"
                | "told"
                | "to"
                | "with"
        ) {
            return false;
        }
        if before.is_empty()
            || before
                .chars()
                .next_back()
                .is_some_and(|character| matches!(character, '.' | '!' | '?' | ';' | ':'))
            || matches!(previous, "and" | "but" | "while")
        {
            return !after.starts_with(',');
        }
        let next = after
            .split(|character: char| !character.is_ascii_alphabetic())
            .find(|value| !value.is_empty())
            .unwrap_or_default();
        matches!(
            next,
            "am" | "are"
                | "became"
                | "began"
                | "can"
                | "could"
                | "did"
                | "does"
                | "gave"
                | "had"
                | "has"
                | "is"
                | "joined"
                | "likes"
                | "loved"
                | "loves"
                | "moved"
                | "painted"
                | "plans"
                | "read"
                | "started"
                | "was"
                | "went"
                | "will"
                | "would"
        )
    })
}

fn statement_has_first_person_reference(statement: &str) -> bool {
    first_person_token_regex().is_match(statement)
        || statement.contains("我")
        || statement.contains("我们")
}

fn statement_binds_speaker_as_fact_subject(statement: &str, addressed_people: &[String]) -> bool {
    // "Organizing a camp in my hometown was awesome": a sentence-initial
    // gerund phrase is the speaker's own action, like an elliptical reply.
    let trimmed = statement.trim_start();
    let offset = statement.len() - trimmed.len();
    let first_word_end = trimmed
        .find(|ch: char| !ch.is_alphanumeric() && ch != '\'')
        .unwrap_or(trimmed.len());
    if addressed_people.is_empty()
        && sentence_initial_gerund(statement, offset, offset + first_word_end)
    {
        return true;
    }
    if statement_has_first_person_reference(statement) {
        return first_person_is_fact_subject(statement);
    }
    let trimmed = statement.trim();
    if trimmed.is_empty() || is_low_information_statement(trimmed) {
        return false;
    }
    // Without a first-person reference, "Glad you got some rest after the
    // drive" is about the addressee, and "That must've been breathtaking!"
    // evaluates the previous turn; neither is the speaker's own fact.
    if second_person_or_reaction_statement(trimmed) {
        return false;
    }
    // An unqualified response such as "Researching adoption agencies" belongs
    // to its speaker. A direct address such as "Thanks, Caroline" does not.
    addressed_people.is_empty()
}

fn second_person_or_reaction_statement(statement: &str) -> bool {
    static SECOND_PERSON: OnceLock<Regex> = OnceLock::new();
    static REACTION: OnceLock<Regex> = OnceLock::new();
    let lower = statement.to_lowercase().replace('’', "'");
    SECOND_PERSON
        .get_or_init(|| {
            Regex::new(r"\b(?:you|your|yours|you're|you've|you'll|you'd|u|ur)\b")
                .expect("second-person regex")
        })
        .is_match(&lower)
        || REACTION
            .get_or_init(|| {
                Regex::new(r"^(?:wow[,!.]?\s+|oh[,!.]?\s+)?(?:that|this|it)(?:'s|\s+is)?\s+(?:must(?:'ve|\s+have)?|sounds?|looks?|seems?)\b")
                    .expect("reaction regex")
            })
            .is_match(&lower)
}

fn first_person_is_fact_subject(statement: &str) -> bool {
    static LEADING_FIRST_PERSON: OnceLock<Regex> = OnceLock::new();
    static COORDINATED_FIRST_PERSON: OnceLock<Regex> = OnceLock::new();
    static EMBEDDED_FIRST_PERSON: OnceLock<Regex> = OnceLock::new();
    let lower = statement.trim().to_ascii_lowercase();
    LEADING_FIRST_PERSON
        .get_or_init(|| {
            Regex::new(r"^(?:i|we|my|our)\b").expect("leading first-person subject regex")
        })
        .is_match(&lower)
        || COORDINATED_FIRST_PERSON
            .get_or_init(|| {
                Regex::new(r"\b(?:and|with)\s+(?:i|we)\b")
                    .expect("coordinated first-person subject regex")
            })
            .is_match(&lower)
        || lower.contains("我")
        || lower.contains("我们")
        || EMBEDDED_FIRST_PERSON
            .get_or_init(|| {
                Regex::new(r"\b(?:i|we)\b|(?:^|[.!?]\s*)(?:my|our)\b|\b(?:is|are|was|were)\s+my\b|\b(?:help|helps|helped|benefits|benefited)\s+me\b")
                    .expect("embedded first-person subject")
            })
            .is_match(&lower)
}

fn atom_matches_person(
    atom: &EvidenceAtom,
    _text_lower: &str,
    person: &str,
    require_speaker: bool,
) -> bool {
    let person_key = normalize_person_key(person);
    if require_speaker && atom.speaker.is_some() {
        if atom
            .subject_binding
            .iter()
            .any(|binding| binding == &format!("explicit_named_subject:{person_key}"))
        {
            return true;
        }
        return atom
            .speaker
            .as_ref()
            .is_some_and(|value| normalize_person_key(value) == person_key);
    }
    atom_has_fact_subject(atom, person)
        || atom
            .mentioned_people
            .iter()
            .any(|value| normalize_person_key(value) == person_key)
        || (atom.speaker.is_none()
            && atom
                .entity
                .as_ref()
                .is_some_and(|value| normalize_person_key(value) == person_key))
}

fn atom_has_fact_subject(atom: &EvidenceAtom, person: &str) -> bool {
    let person_key = normalize_person_key(person);
    atom.fact_subjects
        .iter()
        .any(|value| normalize_person_key(value) == person_key)
        || statement_explicitly_attributes_person(&atom.text, person)
        || (atom.speaker.is_none()
            && atom
                .entity
                .as_ref()
                .is_some_and(|value| normalize_person_key(value) == person_key))
}

/// Final Claim proof may only rely on the durable attribution field. Textual
/// name occurrence is useful for retrieval but cannot silently turn an
/// addressed person or a third-party mention into the fact subject.
fn atom_has_verified_fact_subject(atom: &EvidenceAtom, person: &str) -> bool {
    let person_key = normalize_person_key(person);
    atom.fact_subjects
        .iter()
        .any(|value| normalize_person_key(value) == person_key)
}

fn plan_requests_structured_profile(plan: &QueryPlan) -> bool {
    plan.query_terms
        .iter()
        .chain(plan.object_terms.iter())
        .any(|term| {
            matches!(
                term.as_str(),
                "profile"
                    | "profiles"
                    | "style"
                    | "habit"
                    | "habits"
                    | "preference"
                    | "preferences"
                    | "pattern"
                    | "patterns"
                    | "文风"
                    | "风格"
                    | "习惯"
                    | "画像"
            )
        })
}

fn query_subject_is_action_agent(query_lower: &str, subject: Option<&str>) -> bool {
    let Some(subject) = subject else {
        return false;
    };
    let subject = subject.to_lowercase();
    if subject.ends_with("'s") || subject.ends_with("’s") {
        return false;
    }
    let query = query_lower.trim_start();
    let subject_after_question_verb = ["what", "which", "when", "where", "why", "how", "who"]
        .iter()
        .any(|question_word| {
            query.starts_with(&format!("{question_word} "))
                && query.contains(&format!("did {subject} "))
        });
    subject_tail_after_question_auxiliary(query, &subject).is_some()
        || query_has_household_possessive_subject(query, &subject)
        || subject_after_question_verb
        || [
            format!("what did {subject} "),
            format!("what was {subject} "),
            format!("what does {subject} "),
            format!("what has {subject} "),
            format!("what had {subject} "),
            format!("how did {subject} "),
            format!("when did {subject} "),
            format!("where did {subject} "),
            format!("why did {subject} "),
            format!("which book was {subject} "),
        ]
        .iter()
        .any(|prefix| query.starts_with(prefix))
}

const QUESTION_SUBJECT_AUXILIARIES: &[&str] = &[
    "do", "does", "did", "has", "have", "had", "is", "are", "was", "were", "will",
];

fn subject_tail_after_question_auxiliary<'a>(
    query_lower: &'a str,
    subject: &str,
) -> Option<&'a str> {
    let subject = subject.trim().to_lowercase();
    if subject.is_empty() {
        return None;
    }
    for (start, _) in query_lower.match_indices(&subject) {
        let end = start + subject.len();
        let before = query_lower[..start].chars().next_back();
        let after = query_lower[end..].chars().next();
        if before.is_some_and(|character| character.is_alphanumeric())
            || after.is_some_and(|character| character.is_alphanumeric())
        {
            continue;
        }
        let auxiliary = query_lower[..start]
            .split(|character: char| !character.is_ascii_alphabetic())
            .filter(|value| !value.is_empty())
            .next_back();
        if !auxiliary.is_some_and(|value| QUESTION_SUBJECT_AUXILIARIES.contains(&value)) {
            continue;
        }
        let tail = query_lower[end..].trim_start();
        if tail.starts_with("'s") || tail.starts_with("’s") {
            continue;
        }
        return Some(tail);
    }
    None
}

fn query_has_household_possessive_subject(query_lower: &str, subject: &str) -> bool {
    let subject = subject.trim().to_lowercase();
    if subject.is_empty() {
        return false;
    }
    ["kid", "kids", "child", "children", "family"]
        .iter()
        .any(|dependent| {
            query_lower.contains(&format!("{subject}'s {dependent}"))
                || query_lower.contains(&format!("{subject}’s {dependent}"))
        })
}

fn subject_local_action_terms(query_lower: &str, subject: Option<&str>) -> Vec<String> {
    let Some(subject) = subject else {
        return Vec::new();
    };
    let Some(tail) = subject_tail_after_question_auxiliary(query_lower, subject) else {
        return Vec::new();
    };
    let Some(candidate) = word_regex().find(tail) else {
        return Vec::new();
    };
    let term = candidate.as_str().to_lowercase();
    if matches!(
        term.as_str(),
        "a" | "an"
            | "the"
            | "to"
            | "be"
            | "been"
            | "being"
            | "not"
            | "kind"
            | "of"
            | "more"
            | "most"
            | "really"
            | "very"
            | "and"
            | "or"
            | "but"
            | "as"
            | "at"
            | "by"
            | "for"
            | "from"
            | "in"
            | "on"
            | "with"
            | "about"
            | "before"
            | "after"
            | "during"
            | "while"
            | "get"
            | "got"
            | "ago"
            | "year"
            | "years"
            | "month"
            | "months"
            | "week"
            | "weeks"
            | "day"
            | "days"
    ) {
        return Vec::new();
    }
    (term.len() >= 3
        && term
            .chars()
            .all(|character| character.is_ascii_alphabetic()))
    .then_some(term)
    .into_iter()
    .collect()
}

fn query_requires_subject_speaker(plan: &QueryPlan) -> bool {
    plan.query_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "say"
                | "said"
                | "tell"
                | "told"
                | "describe"
                | "described"
                | "mention"
                | "mentioned"
                | "reply"
                | "replied"
        )
    })
}

fn atom_matches_subject(
    atom: &EvidenceAtom,
    text_lower: &str,
    subject: &str,
    speaker: bool,
) -> bool {
    if speaker && atom.speaker.is_some() {
        if atom.subject_binding.iter().any(|binding| {
            binding == &format!("explicit_named_subject:{}", normalize_person_key(subject))
        }) {
            return true;
        }
        return atom
            .speaker
            .as_ref()
            .is_some_and(|value| normalize_person_key(value) == normalize_person_key(subject));
    }
    let _ = text_lower;
    atom_has_fact_subject(atom, subject)
}

fn term_document_frequencies(atoms: &[EvidenceAtom], terms: &[String]) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    for term in terms {
        let matching_memories = atoms
            .iter()
            .filter(|atom| text_matches_term(&atom.text.to_lowercase(), term))
            .map(|atom| atom.memory_id)
            .collect::<HashSet<_>>();
        frequencies.insert(term.clone(), matching_memories.len());
    }
    frequencies
}

fn lexical_term_document_frequencies(
    atoms: &[EvidenceAtom],
    terms: &[String],
) -> HashMap<String, usize> {
    let mut frequencies = HashMap::new();
    for term in terms {
        let matching_memories = atoms
            .iter()
            .filter(|atom| text_matches_lexical_term(&atom.text.to_lowercase(), term))
            .map(|atom| atom.memory_id)
            .collect::<HashSet<_>>();
        frequencies.insert(term.clone(), matching_memories.len());
    }
    frequencies
}

fn weighted_term_overlap(
    text_lower: &str,
    terms: &[String],
    frequencies: &HashMap<String, usize>,
    document_count: usize,
) -> f32 {
    terms
        .iter()
        .filter(|term| text_matches_term(text_lower, term))
        .map(|term| {
            let frequency = frequencies.get(term).copied().unwrap_or(0) as f32;
            1.0 + (((document_count + 1) as f32) / (frequency + 1.0)).ln()
        })
        .sum()
}

fn weighted_lexical_term_overlap(
    text_lower: &str,
    terms: &[String],
    frequencies: &HashMap<String, usize>,
    document_count: usize,
) -> f32 {
    terms
        .iter()
        .filter(|term| text_matches_lexical_term(text_lower, term))
        .map(|term| {
            let frequency = frequencies.get(term).copied().unwrap_or(0) as f32;
            1.0 + (((document_count + 1) as f32) / (frequency + 1.0)).ln()
        })
        .sum()
}

fn is_low_information_statement(value: &str) -> bool {
    let normalized = normalize_for_dedup(value);
    let token_count = word_regex().find_iter(value).count();
    let lower = value.to_lowercase();
    let acknowledgement_tokens = word_regex()
        .find_iter(&lower)
        .map(|capture| capture.as_str())
        .collect::<Vec<_>>();
    let short_acknowledgement = token_count <= 4
        && acknowledgement_tokens.first().is_some_and(|marker| {
            matches!(
                *marker,
                "hey"
                    | "hi"
                    | "hello"
                    | "thanks"
                    | "thank"
                    | "yep"
                    | "yeah"
                    | "yes"
                    | "definitely"
                    | "sure"
                    | "absolutely"
                    | "indeed"
                    | "okay"
                    | "ok"
            )
        })
        && acknowledgement_tokens.iter().skip(1).all(|token| {
            matches!(
                *token,
                "i" | "do"
                    | "too"
                    | "same"
                    | "yes"
                    | "yeah"
                    | "yep"
                    | "sure"
                    | "thanks"
                    | "thank"
                    | "you"
                    | "of"
                    | "course"
                    | "agree"
                    | "right"
                    | "correct"
            )
        });
    normalized.is_empty()
        || [
            "thatsgreat",
            "thatssogreat",
            "soundsgreat",
            "soundsawesome",
            "thatsoundsawesome",
            "thatsawesome",
            "thatssocool",
            "thatscool",
            "soundsexciting",
            "thatssweet",
            "wow",
            "nice",
            "awesome",
        ]
        .iter()
        .any(|generic| normalized == *generic)
        || short_acknowledgement
        || acknowledgement_with_vocative_regex().is_match(value.trim())
        || content_free_reaction(value)
}

/// "Wow, Calvin!", "Nice job, Jolene!", "Wow that's so cool!": a short reply
/// that has no content once interjections, praise, pronouns, filler verbs and
/// capitalized (addressed) names are removed. "Wow, that's a great necklace!"
/// keeps "necklace" and still counts as content.
fn content_free_reaction(value: &str) -> bool {
    const FILLER: &[&str] = &[
        "wow", "whoa", "oh", "ooh", "aww", "ah", "haha", "lol", "omg", "yay", "nice", "job",
        "great", "awesome", "cool", "amazing", "incredible", "fantastic", "wonderful", "lovely",
        "beautiful", "sweet", "brilliant", "super", "pretty", "good", "glad", "happy", "fun",
        "interesting", "impressive", "congrats", "congratulations", "thanks", "thank", "hey",
        "hi", "hello", "yeah", "yes", "yep", "yup", "sure", "totally", "definitely",
        "absolutely", "really", "so", "such", "very", "too", "that", "that's", "thats", "this",
        "it", "it's", "its", "is", "are", "was", "be", "sounds", "sound", "looks", "look",
        "see", "love", "you", "your", "you're", "me", "i", "i'm", "a", "an", "the", "for",
        "to", "well", "done", "keep", "up", "going", "what", "how", "man", "dude", "buddy",
    ];
    // A name is only filler when it is the addressee: the final word, set off
    // by a comma or following a greeting/thanks ("Definitely Star Wars!" is content).
    const VOCATIVE_LEADS: &[&str] = &["thanks", "thank", "you", "hey", "hi", "hello", "congrats", "congratulations"];
    let tokens = word_regex().find_iter(value).collect::<Vec<_>>();
    let last = tokens.len().saturating_sub(1);
    !tokens.is_empty()
        && tokens.len() <= 8
        && tokens.iter().enumerate().all(|(index, token)| {
            let lower = token.as_str().to_lowercase().replace('’', "'");
            if FILLER.contains(&lower.as_str()) {
                return true;
            }
            index > 0
                && index == last
                && token.as_str().chars().next().is_some_and(char::is_uppercase)
                && (value[..token.start()].trim_end().ends_with(',')
                    || VOCATIVE_LEADS.contains(&tokens[index - 1].as_str().to_lowercase().as_str()))
        })
}

fn acknowledgement_with_vocative_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?i)^\s*(?:thanks?|thank you|hey|hi|hello|yep|yeah|yes|okay|ok|sure|absolutely)(?:\s*,\s*[a-z][a-z'’-]{0,63})?\s*[.!?…。！？]*\s*$",
        )
        .expect("answer acknowledgement regex")
    })
}

fn question_statement_can_fill_answer_slot(plan: &QueryPlan, value: &str) -> bool {
    if !value.contains('?') && !value.contains('？') {
        return true;
    }
    plan.action_terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "ask" | "asked" | "question" | "wonder" | "wondered"
        )
    })
}

fn statement_has_direct_description_value(plan: &QueryPlan, statement_lower: &str) -> bool {
    plan.answer_target == AnswerTarget::Description
        && !plan.object_terms.is_empty()
        && matching_lexical_term_count(statement_lower, &plan.object_terms) > 0
        && contains_any(
            statement_lower,
            &[
                " is ",
                " are ",
                " was ",
                " were ",
                " it's ",
                " that's ",
                " creates a ",
                " creates an ",
                " makes a ",
                " makes an ",
                " feels ",
                " becomes ",
                " looks ",
                " seems ",
            ],
        )
}

fn requires_action_object_complement(
    answer_target: AnswerTarget,
    inference_type: InferenceType,
    action_terms: &[String],
    object_terms: &[String],
) -> bool {
    inference_type == InferenceType::Direct
        && answer_target == AnswerTarget::Object
        && object_terms.is_empty()
        && action_terms.iter().any(|term| {
            term_forms(term).iter().any(|form| {
                !matches!(
                    form.as_str(),
                    "be" | "being"
                        | "do"
                        | "does"
                        | "did"
                        | "go"
                        | "going"
                        | "went"
                        | "happen"
                        | "happened"
                )
            })
        })
}

fn statement_has_action_object_complement(plan: &QueryPlan, statement_lower: &str) -> bool {
    if !requires_action_object_complement(
        plan.answer_target,
        plan.inference_type,
        &plan.action_terms,
        &plan.object_terms,
    ) {
        return true;
    }

    let action_forms = plan
        .action_terms
        .iter()
        .flat_map(|term| term_forms(term))
        .map(|form| normalize_for_dedup(&form))
        .collect::<HashSet<_>>();
    let terms = word_regex()
        .find_iter(statement_lower)
        .map(|value| normalize_for_dedup(value.as_str()))
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    let Some(action_position) = terms.iter().rposition(|term| action_forms.contains(term)) else {
        // A compact reply can supply the object without repeating the verb.
        return true;
    };

    for term in terms.iter().skip(action_position + 1) {
        if matches!(
            term.as_str(),
            "a" | "an"
                | "the"
                | "some"
                | "any"
                | "this"
                | "that"
                | "these"
                | "those"
                | "it"
                | "them"
                | "something"
                | "anything"
                | "everything"
                | "nothing"
                | "thing"
                | "things"
                | "stuff"
                | "to"
                | "of"
                | "and"
                | "or"
                | "but"
                | "just"
                | "really"
                | "very"
                | "more"
                | "again"
                | "also"
                | "still"
                | "up"
                | "out"
                | "away"
                | "back"
                | "off"
                | "do"
                | "does"
                | "did"
                | "doing"
                | "go"
                | "going"
                | "went"
                | "want"
                | "wanted"
                | "need"
                | "needed"
                | "plan"
                | "planned"
                | "try"
                | "trying"
                | "today"
                | "tomorrow"
                | "yesterday"
                | "last"
                | "next"
                | "year"
                | "week"
                | "month"
                | "day"
                | "m"
                | "ve"
        ) {
            continue;
        }
        if matches!(term.as_str(), "on" | "about" | "into" | "for") {
            continue;
        }
        if matches!(
            term.as_str(),
            "at" | "in"
                | "from"
                | "with"
                | "by"
                | "during"
                | "when"
                | "while"
                | "after"
                | "before"
                | "because"
                | "if"
        ) {
            return false;
        }
        return true;
    }
    false
}

fn action_term_is_non_observed_question_action(term: &str) -> bool {
    term_forms(term).iter().any(|form| {
        matches!(
            form.as_str(),
            "ask"
                | "asked"
                | "say"
                | "said"
                | "tell"
                | "told"
                | "explain"
                | "explained"
                | "mention"
                | "mentioned"
                | "share"
                | "shared"
                | "recommend"
                | "recommended"
                | "suggest"
                | "suggested"
                | "advise"
                | "advised"
                | "warn"
                | "warned"
                | "remind"
                | "reminded"
                | "invite"
                | "invited"
                | "encourage"
                | "encouraged"
                | "plan"
                | "planned"
                | "want"
                | "wanted"
                | "hope"
                | "hoped"
                | "intend"
                | "intended"
                | "decide"
                | "decided"
                | "expect"
                | "expected"
                | "consider"
                | "considered"
                | "think"
                | "thought"
                | "believe"
                | "believed"
                | "feel"
                | "felt"
                | "like"
                | "liked"
                | "love"
                | "loved"
                | "enjoy"
                | "enjoyed"
                | "prefer"
                | "preferred"
                | "know"
                | "knew"
                | "promise"
                | "promised"
        )
    })
}

fn requires_observed_action_fact(
    answer_target: AnswerTarget,
    subject_is_action_agent: bool,
    action_terms: &[String],
) -> bool {
    answer_target == AnswerTarget::Object
        && subject_is_action_agent
        && !action_terms.is_empty()
        && !action_terms
            .iter()
            .any(|term| action_term_is_non_observed_question_action(term))
}

fn statement_records_observed_action(statement: &str) -> bool {
    let (_, body) = split_speaker_line(statement);
    let terms = word_regex()
        .find_iter(body)
        .map(|value| normalize_for_dedup(value.as_str()))
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>();
    let Some(first) = terms.first() else {
        return false;
    };
    if matches!(
        first.as_str(),
        "please"
            | "do"
            | "dont"
            | "don"
            | "find"
            | "look"
            | "check"
            | "make"
            | "remember"
            | "consider"
            | "try"
            | "keep"
            | "avoid"
            | "take"
            | "get"
            | "start"
            | "stop"
            | "go"
            | "read"
            | "write"
            | "research"
            | "call"
            | "contact"
            | "use"
            | "ask"
            | "tell"
    ) {
        return false;
    }
    if terms.windows(2).any(|pair| {
        matches!(
            pair,
            [first, second]
                if matches!(
                    first.as_str(),
                    "going" | "plan" | "planned" | "want" | "wanted" | "hope" | "hoped"
                        | "intend" | "intended" | "decide" | "decided" | "expect"
                        | "expected" | "need" | "needed" | "have" | "try" | "trying"
                        | "about"
                ) && second == "to"
        )
    }) {
        return false;
    }
    !terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "should" | "could" | "would" | "might" | "must" | "can" | "may" | "ll"
        )
    })
}

fn action_form_is_topic_only(form: &str) -> bool {
    matches!(
        form,
        "book"
            | "novel"
            | "article"
            | "articles"
            | "trip"
            | "deal"
            | "deals"
            | "sponsorship"
            | "endorsement"
            | "challenge"
            | "difficulty"
            | "struggle"
            | "career-high"
            | "accomplishment"
            | "milestone"
            | "training"
            | "workout"
            | "yoga"
            | "performance"
    )
}

fn statement_has_direct_action_predicate(plan: &QueryPlan, statement_lower: &str) -> bool {
    plan.action_terms.is_empty()
        || plan.action_terms.iter().any(|term| {
            term_forms(term)
                .iter()
                .filter(|form| !action_form_is_topic_only(form))
                .any(|form| text_matches_term_form(statement_lower, form))
        })
}

fn statement_signals_achievement_extremum(statement_lower: &str) -> bool {
    let has_extremum = contains_any(
        statement_lower,
        &[
            "career-high",
            "career high",
            "personal best",
            "personal-best",
            "personal record",
            "record high",
            "record-breaking",
            "record breaking",
            "highest ever",
            "all-time high",
            "职业生涯新高",
            "生涯新高",
            "个人最佳",
            "个人纪录",
            "个人记录",
            "最高纪录",
            "破纪录",
            "刷新纪录",
            "刷新记录",
            "历史最高",
            "最佳成绩",
        ],
    );
    if !has_extremum {
        return false;
    }
    let predicate_markers = [
        "scored ",
        "score ",
        "had a ",
        "had my ",
        "set a ",
        "set my ",
        "reached ",
        "hit ",
        "recorded ",
        "posted ",
        "earned ",
        "achieved ",
        "made ",
        "logged ",
        "got ",
    ];
    predicate_markers.iter().any(|marker| {
        statement_lower.starts_with(marker) || statement_lower.contains(&format!(" {marker}"))
    })
}

fn achievement_extremum_action_term(term: &str) -> bool {
    let term = normalize_for_dedup(term);
    [
        "achieve",
        "achieved",
        "achievement",
        "accomplish",
        "accomplished",
        "record",
        "recorded",
        "set",
        "reach",
        "reached",
        "hit",
        "earn",
        "earned",
        "post",
        "posted",
        "score",
        "scored",
        "make",
        "made",
        "log",
        "logged",
    ]
    .contains(&term.as_str())
}

fn achievement_extremum_object_term(term: &str) -> bool {
    let term = normalize_for_dedup(term);
    [
        "career",
        "high",
        "highest",
        "careerhigh",
        "personalbest",
        "personalrecord",
        "recordhigh",
        "recordbreaking",
        "achievement",
        "achievements",
        "milestone",
        "milestones",
        "performance",
        "performances",
        "record",
        "records",
        "best",
        "peak",
    ]
    .contains(&term.as_str())
}

fn source_level_qualifier_bridge_allowed(plan: &QueryPlan) -> bool {
    plan.cardinality == Cardinality::Single
        && !plan.participant_relationship_qualifier
        && matches!(
            plan.answer_target,
            AnswerTarget::Object | AnswerTarget::Entity | AnswerTarget::Location
        )
}

fn source_qualifier_anchor_matches(
    plan: &QueryPlan,
    candidate: &EvidenceAtom,
    anchor: &EvidenceAtom,
    subject_requires_speaker: bool,
) -> bool {
    let Some(subject) = plan.subject.as_deref() else {
        return false;
    };
    let Some(candidate_speaker) = candidate.speaker.as_deref() else {
        return false;
    };
    if candidate.memory_id != anchor.memory_id
        || candidate.context_group == usize::MAX
        || anchor.context_group == usize::MAX
        || candidate.context_group.abs_diff(anchor.context_group) <= 4
        || !matches!(anchor.kind, EvidenceKind::Statement | EvidenceKind::Event)
        || anchor.kind.grounding_role() != GroundingRole::Original
        || is_low_information_statement(&anchor.text)
    {
        return false;
    }
    let Some(anchor_speaker) = anchor.speaker.as_deref() else {
        return false;
    };
    normalize_person_key(candidate_speaker) == normalize_person_key(anchor_speaker)
        && atom_matches_subject(
            anchor,
            &anchor.text.to_lowercase(),
            subject,
            subject_requires_speaker,
        )
        && plan
            .qualifier_terms
            .iter()
            .all(|term| text_matches_term(&anchor.text.to_lowercase(), term))
}

fn promote_source_qualifier_contexts(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    if !source_level_qualifier_bridge_allowed(plan) || plan.qualifier_terms.is_empty() {
        return;
    }
    let subject_requires_speaker =
        plan.subject_is_action_agent || query_requires_subject_speaker(plan);
    let bridge_facts = atoms
        .iter()
        .filter(|atom| {
            atom_is_answer_bearing(atom)
                && atom
                    .supports_slots
                    .iter()
                    .any(|slot| slot == "source_qualifier_bridge")
        })
        .cloned()
        .collect::<Vec<_>>();
    if bridge_facts.is_empty() {
        return;
    }
    let anchor_ids = atoms
        .iter()
        .filter(|anchor| {
            bridge_facts.iter().any(|candidate| {
                source_qualifier_anchor_matches(plan, candidate, anchor, subject_requires_speaker)
            })
        })
        .map(|anchor| anchor.atom_id.clone())
        .collect::<HashSet<_>>();
    for anchor in atoms
        .iter_mut()
        .filter(|anchor| anchor_ids.contains(&anchor.atom_id))
    {
        let mut slots = anchor
            .supports_slots
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        slots.insert("qualifier_context".to_string());
        slots.insert("source_qualifier_anchor".to_string());
        slots.extend(
            plan.qualifier_terms
                .iter()
                .map(|term| qualifier_slot_name(term)),
        );
        anchor.supports_slots = slots.into_iter().collect();
    }
}

fn participant_relationship_context_matches(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    snapshot: &[EvidenceAtom],
) -> bool {
    if !plan.participant_relationship_qualifier
        || plan.qualifier_terms.is_empty()
        || atom.kind.grounding_role() != GroundingRole::Original
        || atom.context_group == usize::MAX
    {
        return false;
    }
    let Some(speaker) = atom.speaker.as_deref() else {
        return false;
    };
    snapshot.iter().any(|candidate| {
        candidate.memory_id == atom.memory_id
            && candidate.context_group != usize::MAX
            && candidate.context_group.abs_diff(atom.context_group) <= 2
            && candidate.kind.grounding_role() == GroundingRole::Original
            && candidate
                .speaker
                .as_deref()
                .is_some_and(|candidate_speaker| {
                    normalize_person_key(candidate_speaker) == normalize_person_key(speaker)
                })
            && plan
                .qualifier_terms
                .iter()
                .all(|term| text_matches_lexical_term(&candidate.text.to_lowercase(), term))
    })
}

fn is_forward_anaphoric_context(current: &EvidenceAtom, candidate: &EvidenceAtom) -> bool {
    if current.memory_id != candidate.memory_id
        || current.ordinal.saturating_add(1) != candidate.ordinal
        || current.speaker.is_none()
        || current.speaker != candidate.speaker
    {
        return false;
    }

    let lower = candidate.text.trim_start().to_lowercase();
    [
        "it ", "it's ", "its ", "that ", "this ", "they ", "those ", "these ", "and ", "also ",
        "because ", "which ", "who ",
    ]
    .iter()
    .any(|marker| lower.starts_with(marker))
}

fn bounded_list_response_context_bridge(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    adjacent_question: Option<&EvidenceAtom>,
    snapshot: &[EvidenceAtom],
    subject_requires_speaker: bool,
) -> bool {
    if plan.cardinality != Cardinality::List
        || plan.list_scope != "bounded"
        || plan.answer_class_terms.is_empty()
        || atom.kind.grounding_role() != GroundingRole::Original
        || atom.context_group == usize::MAX
        || adjacent_question.is_none()
        || !has_coordinated_proper_value_pair(&atom.text)
    {
        return false;
    }

    let Some(subject) = plan.subject.as_deref() else {
        return false;
    };
    let Some(speaker) = atom.speaker.as_deref() else {
        return false;
    };

    snapshot.iter().any(|candidate| {
        if candidate.memory_id != atom.memory_id
            || candidate.context_group == usize::MAX
            || candidate.context_group >= atom.context_group
            || atom.context_group - candidate.context_group > 3
            || candidate.kind.grounding_role() != GroundingRole::Original
            || is_low_information_statement(&candidate.text)
            || !candidate
                .speaker
                .as_deref()
                .is_some_and(|candidate_speaker| {
                    normalize_person_key(candidate_speaker) == normalize_person_key(speaker)
                })
        {
            return false;
        }

        let candidate_lower = candidate.text.to_lowercase();
        atom_matches_subject(
            candidate,
            &candidate_lower,
            subject,
            subject_requires_speaker,
        ) && plan
            .answer_class_terms
            .iter()
            .all(|term| text_matches_lexical_term(&candidate_lower, term))
            && matching_lexical_term_count(&candidate_lower, &plan.object_terms)
                >= required_object_overlap(plan.object_terms.len())
    })
}

fn initiation_response_context_bridge(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    snapshot: &[EvidenceAtom],
    subject_requires_speaker: bool,
) -> bool {
    if plan.answer_target != AnswerTarget::Initiation
        || plan.object_terms.is_empty()
        || atom.kind.grounding_role() != GroundingRole::Original
        || atom.context_group == usize::MAX
        || !statement_signals_initiation_event(&atom.text.to_lowercase())
    {
        return false;
    }

    let Some(subject) = plan.subject.as_deref() else {
        return false;
    };
    let Some(speaker) = atom.speaker.as_deref() else {
        return false;
    };
    if normalize_person_key(speaker) != normalize_person_key(subject) {
        return false;
    }

    // A direct start event can inherit its activity only from a nearby fact
    // in the same speaker response. This permits "signed me up" after a
    // sentence naming the activity, without letting a distant topic match
    // turn a later opinion into an origin answer.
    snapshot.iter().any(|candidate| {
        if candidate.memory_id != atom.memory_id
            || candidate.context_group == usize::MAX
            || candidate.context_group >= atom.context_group
            || atom.context_group - candidate.context_group > 3
            || candidate.kind.grounding_role() != GroundingRole::Original
            || is_low_information_statement(&candidate.text)
            || !candidate
                .speaker
                .as_deref()
                .is_some_and(|candidate_speaker| {
                    normalize_person_key(candidate_speaker) == normalize_person_key(speaker)
                })
        {
            return false;
        }

        let candidate_lower = candidate.text.to_lowercase();
        atom_matches_subject(
            candidate,
            &candidate_lower,
            subject,
            subject_requires_speaker,
        ) && plan
            .object_terms
            .iter()
            .all(|term| text_matches_lexical_term(&candidate_lower, term))
    })
}

fn has_coordinated_proper_value_pair(value: &str) -> bool {
    coordinated_proper_value_pair_regex().is_match(value)
}

fn statement_signals_media_attribution(statement_lower: &str) -> bool {
    contains_any(
        statement_lower,
        &[
            "picture is from",
            "photo is from",
            "image is from",
            "video is from",
            "audio is from",
            "recording is from",
            "created by",
            "made by",
            "designed by",
            "created the props",
            "创作",
            "制作",
            "来自",
        ],
    )
}

fn media_provenance_response_bridge(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    turn_context_lower: &str,
    snapshot: &[EvidenceAtom],
    subject_requires_speaker: bool,
) -> bool {
    if !plan.media_provenance_requested
        || atom.kind.grounding_role() != GroundingRole::Original
        || !statement_signals_media_attribution(&atom.text.to_lowercase())
    {
        return false;
    }
    let Some(subject) = plan.subject.as_deref() else {
        return false;
    };
    if !atom_matches_subject(
        atom,
        &atom.text.to_lowercase(),
        subject,
        subject_requires_speaker,
    ) {
        return false;
    }
    let named_context_bound = plan.named_entities.is_empty()
        || plan
            .named_entities
            .iter()
            .all(|entity| text_matches_named_entity(turn_context_lower, entity));
    if !named_context_bound {
        return false;
    }
    let has_nearby_media = snapshot.iter().any(|candidate| {
        candidate.memory_id == atom.memory_id
            && candidate.kind == EvidenceKind::NativeMedia
            && candidate.context_group != usize::MAX
            && atom.context_group != usize::MAX
            && candidate.context_group.abs_diff(atom.context_group) <= 4
            && candidate.speaker.as_deref().is_some_and(|speaker| {
                normalize_person_key(speaker) == normalize_person_key(subject)
            })
    });
    let has_object_context = plan.object_terms.is_empty()
        || snapshot.iter().any(|candidate| {
            candidate.memory_id == atom.memory_id
                && candidate.kind.grounding_role() == GroundingRole::Original
                && candidate.context_group != usize::MAX
                && atom.context_group != usize::MAX
                && candidate.context_group.abs_diff(atom.context_group) <= 4
                && plan
                    .object_terms
                    .iter()
                    .any(|term| text_matches_lexical_term(&candidate.text.to_lowercase(), term))
        });
    has_nearby_media && has_object_context
}

fn shared_event_activity_value(value: &str) -> Option<String> {
    let lower = value.to_lowercase();
    for marker in [
        "ended up in ",
        "ended up playing ",
        "played ",
        "participated in ",
        "took part in ",
        "competed in ",
    ] {
        let Some(start) = lower.find(marker) else {
            continue;
        };
        let tail = value[start + marker.len()..]
            .split(['.', '!', '?', ';'])
            .next()
            .unwrap_or_default()
            .trim();
        if !tail.is_empty() {
            return Some(tail.to_string());
        }
    }
    None
}

fn co_participant_event_bridge(plan: &QueryPlan, atom: &EvidenceAtom) -> bool {
    if !plan.co_participant_event_requested
        || atom.kind.grounding_role() != GroundingRole::Original
        || plan.people.len() < 2
        || shared_event_activity_value(&atom.text).is_none()
    {
        return false;
    }
    plan.people
        .iter()
        .all(|person| atom_has_verified_fact_subject(atom, person))
}

fn native_media_is_answer_preferred(plan: &QueryPlan) -> bool {
    plan.requested_modalities
        .iter()
        .any(|modality| modality != "text")
        || (matches!(
            plan.answer_target,
            AnswerTarget::Object | AnswerTarget::Entity
        ) && plan
            .action_terms
            .iter()
            .any(|action| matches!(action.as_str(), "give" | "share" | "show" | "take")))
}

fn extract_location_answer_value(plan: &QueryPlan, value: &str) -> Option<String> {
    for capture in location_value_regex()
        .captures_iter(value)
        .chain(answer_first_location_value_regex().captures_iter(value))
        .chain(named_location_value_regex().captures_iter(value))
    {
        let Some(location) = capture.get(1).map(|value| value.as_str().trim()) else {
            continue;
        };
        let normalized = normalize_person_key(location);
        if normalized.is_empty()
            || plan
                .people
                .iter()
                .any(|person| normalize_person_key(person) == normalized)
            || plan
                .subject
                .as_ref()
                .is_some_and(|subject| normalize_person_key(subject) == normalized)
        {
            continue;
        }
        let matched = capture
            .get(0)
            .map(|value| value.as_str().to_ascii_lowercase())
            .unwrap_or_default();
        let location_lower = location.to_ascii_lowercase();
        return Some(if matched.contains(&format!("the {location_lower}")) {
            format!("the {location}")
        } else {
            location.to_string()
        });
    }
    None
}

fn atom_has_location_value(plan: &QueryPlan, value: &str) -> bool {
    location_value_regex()
        .captures_iter(value)
        .chain(answer_first_location_value_regex().captures_iter(value))
        .chain(named_location_value_regex().captures_iter(value))
        .filter_map(|capture| capture.get(1))
        .map(|capture| capture.as_str().trim())
        .any(|candidate| {
            let normalized = normalize_person_key(candidate);
            !normalized.is_empty()
                && !plan
                    .people
                    .iter()
                    .any(|person| normalize_person_key(person) == normalized)
                && !plan
                    .subject
                    .as_ref()
                    .is_some_and(|subject| normalize_person_key(subject) == normalized)
        })
}

fn location_presence_term(term: &str) -> bool {
    matches!(
        term,
        "am" | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "been"
            | "travel"
            | "traveling"
            | "travelling"
            | "trip"
            | "visit"
            | "visiting"
            | "visited"
            | "go"
            | "went"
            | "gone"
            | "stay"
            | "stayed"
            | "live"
            | "lived"
            | "move"
            | "moved"
            | "arrive"
            | "arrived"
            | "head"
            | "headed"
            | "fly"
            | "flew"
            | "drive"
            | "drove"
            | "return"
            | "returned"
            | "leave"
            | "left"
    )
}

fn location_query_requires_subject_presence(plan: &QueryPlan) -> bool {
    plan.answer_target == AnswerTarget::Location
        && plan.subject.is_some()
        && (plan.action_terms.is_empty()
            || plan.action_terms.iter().any(|term| {
                term_forms(term)
                    .iter()
                    .any(|form| location_presence_term(form))
            }))
}

fn statement_has_first_person_location_presence(statement_lower: &str) -> bool {
    let words = word_regex()
        .find_iter(statement_lower)
        .map(|value| value.as_str().to_ascii_lowercase())
        .collect::<Vec<_>>();
    words.windows(2).any(|pair| {
        matches!(pair[0].as_str(), "i" | "im" | "ive" | "we") && location_presence_term(&pair[1])
    }) || words.windows(3).any(|triple| {
        matches!(triple[0].as_str(), "my" | "our")
            && matches!(triple[1].as_str(), "trip" | "visit" | "travel")
            && matches!(
                triple[2].as_str(),
                "to" | "in" | "through" | "around" | "from"
            )
    })
}

fn statement_has_named_subject_location_presence(statement_lower: &str, subject: &str) -> bool {
    let subject_words = word_regex()
        .find_iter(&subject.to_ascii_lowercase())
        .map(|value| value.as_str().to_ascii_lowercase())
        .collect::<Vec<_>>();
    if subject_words.is_empty() {
        return false;
    }
    let statement_words = word_regex()
        .find_iter(statement_lower)
        .map(|value| value.as_str().to_ascii_lowercase())
        .collect::<Vec<_>>();
    statement_words
        .windows(subject_words.len().saturating_add(1))
        .any(|window| {
            window[..subject_words.len()]
                .iter()
                .zip(&subject_words)
                .all(|(candidate, expected)| candidate == expected)
                && location_presence_term(&window[subject_words.len()])
        })
}

fn atom_has_location_answer_value(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    statement_lower: &str,
) -> bool {
    if !atom_has_location_value(plan, &atom.text) {
        return false;
    }
    if !location_query_requires_subject_presence(plan) {
        return true;
    }
    let Some(subject) = plan.subject.as_deref() else {
        return true;
    };
    let speaker_matches_subject = atom
        .speaker
        .as_deref()
        .is_some_and(|speaker| normalize_person_key(speaker) == normalize_person_key(subject));
    (speaker_matches_subject && statement_has_first_person_location_presence(statement_lower))
        || statement_has_named_subject_location_presence(statement_lower, subject)
}

/// Slots that express "this passage answers what was asked": a confident
/// reranker verdict covers them. Attribution, time, named entities, list
/// completeness, negation and conditions stay rule-checked.
const SEMANTIC_ANSWER_SLOTS: &[&str] = &[
    "answer_bearing_fact",
    "action",
    "object",
    "answer_class",
    "qualifier_context",
];


/// Semantic relevance over the lexically ranked candidates. The best turns
/// (bounded by `rerank_budget`) are scored with their preceding turn as
/// context. A confident "answers the question" grants the answer, action and
/// object slots that paraphrases fail lexically; a confident "does not"
/// revokes them. Subject attribution is never granted here: an atom must
/// already be attributed to the requested subject. Returns the backend id and
/// the number of scored turns, or `None` when no scorer ran.
fn apply_semantic_relevance(
    question: &str,
    plan: &QueryPlan,
    atoms: &mut [EvidenceAtom],
    scorer: Option<&dyn crate::semantic_scorer::SemanticScorer>,
) -> Option<(String, usize)> {
    use crate::semantic_scorer::{grant_threshold, rerank_budget, revoke_threshold};
    let scorer = scorer?;
    if crate::answer_execution::answer_worker_cancelled() {
        return None;
    }
    // Turn key: dialogue turns group their sentences; other sources score
    // atom by atom.
    let turn_key = |atom: &EvidenceAtom| {
        (
            atom.memory_id,
            if atom.context_group == usize::MAX {
                usize::MAX - 1 - atom.ordinal
            } else {
                atom.context_group
            },
        )
    };
    let scorable = |atom: &EvidenceAtom| {
        atom.kind.grounding_role() == GroundingRole::Original
            && matches!(
                atom.kind,
                EvidenceKind::Statement | EvidenceKind::Event | EvidenceKind::NativeMedia
            )
    };
    let mut turn_text = HashMap::<(u64, usize), (Option<String>, Vec<(usize, String)>)>::new();
    let mut turn_best = HashMap::<(u64, usize), f32>::new();
    for atom in atoms.iter().filter(|atom| scorable(atom)) {
        let key = turn_key(atom);
        let entry = turn_text.entry(key).or_insert_with(|| (atom.speaker.clone(), Vec::new()));
        entry.1.push((atom.ordinal, atom.text.clone()));
        let best = turn_best.entry(key).or_insert(f32::NEG_INFINITY);
        *best = best.max(atom.score);
    }
    let line = |key: &(u64, usize)| {
        turn_text.get(key).map(|(speaker, parts)| {
            let mut parts = parts.clone();
            parts.sort_by_key(|(ordinal, _)| *ordinal);
            let body = parts
                .into_iter()
                .map(|(_, text)| text)
                .collect::<Vec<_>>()
                .join(" ");
            match speaker {
                Some(speaker) => format!("{speaker}: {body}"),
                None => body,
            }
        })
    };
    let mut ranked = turn_best.into_iter().collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    // Lexical rank alone misses a third of evidence turns (replies phrased
    // without the question's words: "Writing and hanging with friends!" for
    // "What are Joanna's hobbies?"). Measured on the LoCoMo dev split, adding
    // the turns most similar by embedding lifts gold-turn coverage from 75% to
    // 86% at 48 + 48 turns, versus 80% for 96 lexical turns.
    let lexical_budget = rerank_budget();
    let mut keys = ranked.iter().take(lexical_budget).map(|(key, _)| *key).collect::<Vec<_>>();
    let dense_budget = crate::semantic_scorer::rerank_dense_budget();
    if dense_budget > 0 && ranked.len() > lexical_budget {
        if let Some(index) = crate::semantic_retrieval::global_index() {
            let rest = ranked.iter().skip(lexical_budget).map(|(key, _)| *key).collect::<Vec<_>>();
            let texts = rest
                .iter()
                .map(|key| {
                    let current = line(key).unwrap_or_default();
                    match (key.1 != 0 && key.1 < usize::MAX / 2).then(|| line(&(key.0, key.1 - 1))).flatten() {
                        Some(previous) => format!("{previous}\n{current}"),
                        None => current,
                    }
                })
                .collect::<Vec<_>>();
            if let Some(similarities) =
                index.passage_similarities(question, &texts.iter().map(String::as_str).collect::<Vec<_>>())
            {
                let mut dense = rest.into_iter().zip(similarities).collect::<Vec<_>>();
                dense.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal).then_with(|| left.0.cmp(&right.0)));
                keys.extend(dense.into_iter().take(dense_budget).map(|(key, _)| key));
            }
        }
    }
    let passages = keys
        .iter()
        .map(|key| {
            let current = line(key).unwrap_or_default();
            let previous = (key.1 != 0 && key.1 < usize::MAX / 2)
                .then(|| line(&(key.0, key.1 - 1)))
                .flatten();
            match previous {
                Some(previous) => format!("{previous}\n{current}"),
                None => current,
            }
        })
        .collect::<Vec<_>>();
    if passages.is_empty() {
        return Some((scorer.backend_id().to_string(), 0));
    }
    let scores = match scorer.relevance(
        question,
        &passages.iter().map(String::as_str).collect::<Vec<_>>(),
    ) {
        Ok(scores) if scores.len() == keys.len() => scores,
        Ok(_) | Err(_) => return None,
    };
    let relevance = keys.iter().copied().zip(scores).collect::<HashMap<_, _>>();
    // A relevant turn usually also holds greetings and reactions ("Sorry to
    // hear that!"), and a long multi-topic turn dilutes its one answering
    // sentence ("...school, fantasy books... I've also started learning the
    // piano..." scores ~0.4 as a whole). Sentences of every turn that is at
    // least plausibly relevant are judged on their own (with the preceding
    // turn as context), so exactly the answer-bearing sentence gets the slots.
    let sentence_rescore_floor = crate::semantic_scorer::sentence_rescore_floor();
    const SENTENCE_RESCORE_TURNS: usize = 12;
    let mut granted_turns = relevance
        .iter()
        .filter(|(_, score)| **score >= sentence_rescore_floor)
        .map(|(key, score)| (*key, *score))
        .collect::<Vec<_>>();
    granted_turns.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(Ordering::Equal));
    granted_turns.truncate(SENTENCE_RESCORE_TURNS);
    let granted_turns = granted_turns
        .into_iter()
        .map(|(key, _)| key)
        .collect::<HashSet<_>>();
    let sentences = atoms
        .iter()
        .filter(|atom| scorable(atom) && granted_turns.contains(&turn_key(atom)))
        .filter(|atom| {
            atoms
                .iter()
                .filter(|other| scorable(other) && turn_key(other) == turn_key(atom))
                .count()
                > 1
        })
        .map(|atom| {
            let key = turn_key(atom);
            let sentence = match &atom.speaker {
                Some(speaker) => format!("{speaker}: {}", atom.text),
                None => atom.text.clone(),
            };
            let previous = (key.1 != 0 && key.1 < usize::MAX / 2)
                .then(|| line(&(key.0, key.1 - 1)))
                .flatten();
            (
                atom.atom_id.clone(),
                match previous {
                    Some(previous) => format!("{previous}\n{sentence}"),
                    None => sentence,
                },
            )
        })
        .collect::<Vec<_>>();
    let sentence_relevance = if sentences.is_empty() {
        HashMap::new()
    } else {
        match scorer.relevance(
            question,
            &sentences.iter().map(|(_, text)| text.as_str()).collect::<Vec<_>>(),
        ) {
            Ok(scores) if scores.len() == sentences.len() => sentences
                .iter()
                .map(|(atom_id, _)| atom_id.clone())
                .zip(scores)
                .collect::<HashMap<_, _>>(),
            Ok(_) | Err(_) => return None,
        }
    };
    // Context may disambiguate an elliptical answer ("Definitely Star Wars!"
    // after "Any favorite movies?") but cannot supply it: a reaction such as
    // "That must've been breathtaking!" scores high only because the previous
    // turn answers the question. Sentences about to be granted are therefore
    // also judged without context; relevance that exists only in the context
    // is uncertain, not confident.
    let contextual_score = |atom: &EvidenceAtom| {
        relevance.get(&turn_key(atom)).map(|turn_score| {
            sentence_relevance
                .get(&atom.atom_id)
                .copied()
                .unwrap_or(*turn_score)
        })
    };
    let standalone = atoms
        .iter()
        .filter(|atom| scorable(atom))
        .filter(|atom| contextual_score(atom).is_some_and(|score| score >= grant_threshold()))
        .map(|atom| {
            (
                atom.atom_id.clone(),
                match &atom.speaker {
                    Some(speaker) => format!("{speaker}: {}", atom.text),
                    None => atom.text.clone(),
                },
            )
        })
        .collect::<Vec<_>>();
    let standalone_relevance = if standalone.is_empty() {
        HashMap::new()
    } else {
        match scorer.relevance(
            question,
            &standalone.iter().map(|(_, text)| text.as_str()).collect::<Vec<_>>(),
        ) {
            Ok(scores) if scores.len() == standalone.len() => standalone
                .iter()
                .map(|(atom_id, _)| atom_id.clone())
                .zip(scores)
                .collect::<HashMap<_, _>>(),
            Ok(_) | Err(_) => return None,
        }
    };
    let context_only = (grant_threshold() - 0.01).max(revoke_threshold());
    // Session dates per record, so a granted atom of a date question can carry
    // `event_time` when its own time expression resolves by the date rules.
    let session_dates = atoms
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::SessionDate)
        .filter_map(|atom| {
            crate::memory_engine_summary::source_session_date(&atom.text).map(|date| (atom.memory_id, date))
        })
        .collect::<HashMap<_, _>>();
    for atom in atoms.iter_mut().filter(|atom| scorable(atom)) {
        let Some(mut score) = contextual_score(atom) else {
            continue;
        };
        if standalone_relevance
            .get(&atom.atom_id)
            .is_some_and(|alone| *alone < revoke_threshold())
        {
            score = score.min(context_only);
        }
        atom.semantic_relevance = Some(score);
        let answer_bearing = atom
            .supports_slots
            .iter()
            .any(|slot| slot == "answer_bearing_fact");
        if score >= grant_threshold() {
            let attributed = plan
                .subject
                .as_deref()
                .is_none_or(|subject| atom_has_verified_fact_subject(atom, subject));
            if attributed
                && !is_low_information_statement(&atom.text)
                && question_statement_can_fill_answer_slot(plan, &atom.text)
            {
                // Confident relevance outranks any lexical boost, and within
                // the confident tier relevance decides: lexical overlap is
                // only a tie-breaker. Otherwise "Researching adoption
                // agencies" (0.85, many shared words) beats the sentence that
                // answers "Why did Caroline choose the agency?" (0.997).
                atom.score = crate::semantic_scorer::relevance_tier() + 10.0 * score + 0.1 * atom.score;
                for slot in SEMANTIC_ANSWER_SLOTS {
                    if !atom.supports_slots.iter().any(|existing| existing == slot) {
                        atom.supports_slots.push(slot.to_string());
                    }
                }
                // The verdict covered the question's qualifiers as a whole.
                for term in &plan.qualifier_terms {
                    atom.supports_slots.push(qualifier_slot_name(term));
                }
                // Relevance says this is the event; the date itself must still
                // be stated or resolvable ("five years ago" + session date).
                let lower = atom.text.to_lowercase();
                if plan.answer_target == AnswerTarget::Temporal
                    && (contains_date_like_value(&lower)
                        || contains_bare_calendar_month_reference(&lower)
                        || session_dates.get(&atom.memory_id).is_some_and(|date| {
                            crate::memory_engine_summary::relative_event_date_answer(&atom.text, *date).is_some()
                        }))
                {
                    atom.supports_slots.push("event_time".to_string());
                    atom.supports_slots.push("temporal_scope".to_string());
                }
                atom.supports_slots.push("semantic_relevance".to_string());
                atom.supports_slots.sort();
                atom.supports_slots.dedup();
            } else {
                atom.score += score;
            }
        } else if score < revoke_threshold()
            // A typed attribute value stated in the source ("tough as a single
            // parent" for a relationship status) is extracted by rule; the
            // reranker judges whole-turn topicality and misses such values.
            && !atom
                .supports_slots
                .iter()
                .any(|slot| slot == "categorical_attribute_value")
        {
            let _ = answer_bearing;
            atom.score -= crate::semantic_scorer::relevance_tier();
            atom.supports_slots.retain(|slot| {
                !SEMANTIC_ANSWER_SLOTS.contains(&slot.as_str()) && slot != "constrained_backoff_fact"
            });
        } else {
            atom.score += score - grant_threshold();
        }
    }
    Some((scorer.backend_id().to_string(), keys.len()))
}

fn rank_evidence_atoms(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    if crate::answer_execution::answer_worker_cancelled() { return; }
    let snapshot = atoms.to_vec();
    let session_dates = snapshot
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::SessionDate)
        .filter_map(|atom| {
            crate::memory_engine_summary::source_session_date(&atom.text)
                .map(|date| (atom.memory_id, date))
        })
        .collect::<HashMap<_, _>>();
    // A source can truthfully provide only a month and year. Keep that
    // precision separate from NaiveDate so no artificial day is introduced.
    let session_month_years = snapshot
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::SessionDate)
        .filter_map(|atom| {
            source_session_month_year(&atom.text).map(|value| (atom.memory_id, value))
        })
        .collect::<HashMap<_, _>>();
    let document_count = snapshot
        .iter()
        .map(|atom| atom.memory_id)
        .collect::<HashSet<_>>()
        .len()
        .max(1);
    let query_frequencies = term_document_frequencies(&snapshot, &plan.query_terms);
    let action_frequencies = term_document_frequencies(&snapshot, &plan.action_terms);
    let object_frequencies = lexical_term_document_frequencies(&snapshot, &plan.object_terms);
    let qualifier_frequencies = lexical_term_document_frequencies(&snapshot, &plan.qualifier_terms);
    let action_threshold = required_action_overlap(plan.action_terms.len());
    let object_threshold = required_object_overlap(plan.object_terms.len());
    let query_event_scope_values = crate::memory_engine_summary::event_values(
        &[
            plan.action_terms.join(" "),
            plan.object_terms.join(" "),
            plan.answer_class_terms.join(" "),
        ]
        .join(" "),
    );
    let mut turn_contexts = HashMap::<(u64, usize), String>::new();
    for candidate in &snapshot {
        if candidate.context_group == usize::MAX {
            continue;
        }
        let context = turn_contexts
            .entry((candidate.memory_id, candidate.context_group))
            .or_default();
        if !context.is_empty() {
            context.push(' ');
        }
        context.push_str(&candidate.text.to_lowercase());
    }
    let atom_by_position = snapshot
        .iter()
        .map(|candidate| ((candidate.memory_id, candidate.ordinal), candidate))
        .collect::<HashMap<_, _>>();
    let mut dialogue_lines = HashMap::<(u64, usize), String>::new();
    for candidate in &snapshot {
        if candidate.context_group == usize::MAX
            || !matches!(
                candidate.kind,
                EvidenceKind::Statement | EvidenceKind::Event
            )
        {
            continue;
        }
        let line = dialogue_lines
            .entry((candidate.memory_id, candidate.context_group))
            .or_default();
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(&candidate.text.to_lowercase());
    }
    let mut nearby_dialogue_contexts = HashMap::<(u64, usize), String>::new();
    for &(memory_id, context_group) in dialogue_lines.keys() {
        let start = context_group.saturating_sub(4);
        let end = context_group.saturating_add(4);
        let nearby = (start..=end)
            .filter_map(|group| dialogue_lines.get(&(memory_id, group)))
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        nearby_dialogue_contexts.insert((memory_id, context_group), nearby);
    }
    // Source-level qualifier anchors depend only on the anchor atom and the
    // plan, so they are resolved once per pass instead of once per
    // (atom, anchor) pair: memory -> [(context group, speaker key)].
    let qualifier_anchors = {
        let mut anchors = HashMap::<u64, Vec<(usize, String)>>::new();
        let subject_requires_speaker =
            plan.subject_is_action_agent || query_requires_subject_speaker(plan);
        if let Some(subject) = plan.subject.as_deref().filter(|_| {
            !plan.qualifier_terms.is_empty() && source_level_qualifier_bridge_allowed(plan)
        }) {
            for anchor in &snapshot {
                let Some(speaker) = anchor.speaker.as_deref() else {
                    continue;
                };
                if anchor.context_group == usize::MAX
                    || !matches!(anchor.kind, EvidenceKind::Statement | EvidenceKind::Event)
                    || anchor.kind.grounding_role() != GroundingRole::Original
                    || is_low_information_statement(&anchor.text)
                {
                    continue;
                }
                let anchor_lower = anchor.text.to_lowercase();
                if atom_matches_subject(anchor, &anchor_lower, subject, subject_requires_speaker)
                    && plan
                        .qualifier_terms
                        .iter()
                        .all(|term| text_matches_term(&anchor_lower, term))
                {
                    anchors
                        .entry(anchor.memory_id)
                        .or_default()
                        .push((anchor.context_group, normalize_person_key(speaker)));
                }
            }
        }
        anchors
    };
    // Cross-record qualifier candidates, likewise resolved once per pass:
    // speaker key -> memories holding a qualifying statement by that speaker.
    let cross_record_qualifiers = {
        let mut candidates = HashMap::<String, HashSet<u64>>::new();
        let subject_requires_speaker =
            plan.subject_is_action_agent || query_requires_subject_speaker(plan);
        if let Some(subject) = plan.subject.as_deref().filter(|_| {
            !plan.qualifier_terms.is_empty()
                && source_level_qualifier_bridge_allowed(plan)
                && plan.subject_is_action_agent
        }) {
            for candidate in &snapshot {
                let Some(speaker) = candidate.speaker.as_deref() else {
                    continue;
                };
                if !matches!(candidate.kind, EvidenceKind::Statement | EvidenceKind::Event)
                    || candidate.kind.grounding_role() != GroundingRole::Original
                    || is_low_information_statement(&candidate.text)
                {
                    continue;
                }
                let candidate_lower = candidate.text.to_lowercase();
                if atom_matches_subject(candidate, &candidate_lower, subject, subject_requires_speaker)
                    && plan
                        .qualifier_terms
                        .iter()
                        .all(|term| text_matches_lexical_term(&candidate_lower, term))
                {
                    candidates
                        .entry(normalize_person_key(speaker))
                        .or_default()
                        .insert(candidate.memory_id);
                }
            }
        }
        candidates
    };
    for atom in atoms.iter_mut() {
        if crate::answer_execution::answer_worker_cancelled() { return; }
        let lower = atom.text.to_lowercase();
        let previous = atom
            .ordinal
            .checked_sub(1)
            .and_then(|ordinal| atom_by_position.get(&(atom.memory_id, ordinal)).copied());
        let adjacent_question = previous
            .filter(|candidate| candidate.text.contains('?') || candidate.text.contains('？'))
            .or_else(|| {
                // Media captions may be projected between a question and its
                // reply. Recover only the immediately preceding dialogue
                // question, rather than treating the whole source as context.
                snapshot
                    .iter()
                    .filter(|candidate| {
                        candidate.memory_id == atom.memory_id
                            && candidate.context_group != usize::MAX
                            && atom.context_group != usize::MAX
                            && candidate.context_group < atom.context_group
                            && atom.context_group - candidate.context_group <= 1
                            && matches!(
                                candidate.kind,
                                EvidenceKind::Statement | EvidenceKind::Event
                            )
                            && (candidate.text.contains('?') || candidate.text.contains('？'))
                    })
                    .max_by_key(|candidate| (candidate.context_group, candidate.ordinal))
            });
        let forward_anaphora = atom_by_position
            .get(&(atom.memory_id, atom.ordinal.saturating_add(1)))
            .copied()
            .filter(|candidate| is_forward_anaphoric_context(atom, candidate));
        let turn_context_lower = turn_contexts
            .get(&(atom.memory_id, atom.context_group))
            .cloned()
            .unwrap_or_else(|| lower.clone());
        let contextual_lower = [
            adjacent_question.map(|candidate| candidate.text.to_lowercase()),
            Some(turn_context_lower.clone()),
            Some(lower.clone()),
            forward_anaphora.map(|candidate| candidate.text.to_lowercase()),
        ]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>()
        .join(" ");
        let query_weight = weighted_term_overlap(
            &contextual_lower,
            &plan.query_terms,
            &query_frequencies,
            document_count,
        );
        let action_weight = weighted_term_overlap(
            &contextual_lower,
            &plan.action_terms,
            &action_frequencies,
            document_count,
        );
        let object_weight = weighted_lexical_term_overlap(
            &contextual_lower,
            &plan.object_terms,
            &object_frequencies,
            document_count,
        );
        let qualifier_weight = weighted_lexical_term_overlap(
            &contextual_lower,
            &plan.qualifier_terms,
            &qualifier_frequencies,
            document_count,
        );
        let query_overlap = matching_term_count(&contextual_lower, &plan.query_terms);
        let action_overlap = matching_term_count(&contextual_lower, &plan.action_terms);
        let atom_event_scope_values = crate::memory_engine_summary::event_values(&lower);
        let semantic_event_scope_bound = matches!(
            plan.answer_target,
            AnswerTarget::Temporal | AnswerTarget::Event
        ) && !query_event_scope_values.is_empty()
            && atom_event_scope_values
                .iter()
                .any(|value| query_event_scope_values.contains(value));
        let initiation_event_bound = plan.answer_target == AnswerTarget::Initiation
            && statement_signals_initiation_event(&lower);
        let achievement_extremum_bound =
            plan.achievement_extremum && statement_signals_achievement_extremum(&lower);
        let achievement_action_bound = achievement_extremum_bound
            && plan
                .action_terms
                .iter()
                .all(|term| achievement_extremum_action_term(term));
        let action_bound = action_threshold == 0
            || action_overlap >= action_threshold
            || initiation_event_bound
            || achievement_action_bound
            || semantic_event_scope_bound;
        let direct_action_bound = statement_has_direct_action_predicate(plan, &lower)
            || achievement_action_bound
            || semantic_event_scope_bound;
        let action_object_complement_required = requires_action_object_complement(
            plan.answer_target,
            plan.inference_type,
            &plan.action_terms,
            &plan.object_terms,
        );
        let action_object_complement_bound = !action_object_complement_required
            || statement_has_action_object_complement(plan, &lower);
        let observed_action_fact_required = requires_observed_action_fact(
            plan.answer_target,
            plan.subject_is_action_agent,
            &plan.action_terms,
        );
        let co_participant_event = co_participant_event_bridge(plan, atom);
        let observed_action_directness_bound = !observed_action_fact_required
            || direct_action_bound
            || co_participant_event
            || (plan.answer_shape == AnswerShape::ActionSet
                && strategy_statement_has_concrete_action(&atom.text, atom.speaker.as_deref()))
            || adjacent_question.is_some_and(|question| {
                statement_has_direct_action_predicate(plan, &question.text.to_lowercase())
            });
        let observed_action_fact_bound = !observed_action_fact_required
            || (observed_action_directness_bound && statement_records_observed_action(&atom.text));
        let categorical_attribute_value_bound =
            plan.categorical_attribute_kind.is_none_or(|kind| {
                atom.kind.grounding_role() == GroundingRole::Original
                    && crate::memory_engine_summary::categorical_value(kind, &atom.text).is_some()
                    && plan
                        .subject
                        .as_ref()
                        .is_none_or(|subject| atom_matches_subject(atom, &lower, subject, false))
            });
        let duration_value_required = plan.answer_target == AnswerTarget::Duration;
        let duration_value_bound = !duration_value_required
            || (atom.kind.grounding_role() == GroundingRole::Original
                && crate::memory_engine_summary::source_has_duration_value(&atom.text));
        let (list_values, same_turn_list_value_bridge) =
            atom_grounded_list_values_with_same_turn_context(plan, atom, &turn_context_lower);
        let list_value_candidate =
            atom.kind.grounding_role() == GroundingRole::Original && !list_values.is_empty();
        let typed_location_list_value_bound = plan.grounded_list_kind
            == Some(crate::memory_engine_summary::GroundedListKind::Locations)
            && list_value_candidate;
        let location_value_required = plan.answer_target == AnswerTarget::Location;
        let location_value_bound = !location_value_required
            || typed_location_list_value_bound
            || atom_has_location_answer_value(plan, atom, &lower);
        let household_dependent_preference_bound = !plan.household_dependent_preference
            || (atom.kind.grounding_role() == GroundingRole::Original
                && household_dependent_preference_value(plan, atom, &snapshot));

        let subject_requires_speaker =
            plan.subject_is_action_agent || query_requires_subject_speaker(plan);
        let bounded_list_response_bridge = bounded_list_response_context_bridge(
            plan,
            atom,
            adjacent_question,
            &snapshot,
            subject_requires_speaker,
        );
        let initiation_response_bridge = initiation_event_bound
            && initiation_response_context_bridge(plan, atom, &snapshot, subject_requires_speaker);
        let object_overlap = matching_lexical_term_count(&contextual_lower, &plan.object_terms);
        let exact_object_overlap = plan
            .object_terms
            .iter()
            .filter(|term| text_matches_term_form(&contextual_lower, term))
            .count();
        let achievement_object_bound = achievement_extremum_bound
            && plan
                .object_terms
                .iter()
                .filter(|term| !achievement_extremum_object_term(term))
                .all(|term| text_matches_lexical_term(&contextual_lower, term));
        // Summary/profile records commonly encode the requested field as a
        // structural speaker label (`Summary:`). The label is removed from
        // the body statement by projection, so treat that typed label as the
        // object match without making arbitrary speaker names satisfy object
        // slots.
        let summary_label_bound = plan.object_terms.iter().any(|term| {
            matches!(
                term.as_str(),
                "summary" | "summaries" | "overview" | "synopsis" | "概要" | "摘要"
            )
        }) && atom
            .speaker
            .as_deref()
            .is_some_and(|speaker| normalize_person_key(speaker) == "summary");
        let object_bound = object_threshold == 0
            || object_overlap >= object_threshold
            || bounded_list_response_bridge
            || initiation_response_bridge
            || achievement_object_bound
            || semantic_event_scope_bound
            || summary_label_bound
            || (plan.categorical_attribute_kind.is_some() && categorical_attribute_value_bound);
        let direct_description_value = statement_has_direct_description_value(plan, &lower);
        let answer_class_overlap =
            matching_lexical_term_count(&contextual_lower, &plan.answer_class_terms);
        let achievement_answer_class_bound = achievement_extremum_bound
            && plan
                .answer_class_terms
                .iter()
                .all(|term| achievement_extremum_object_term(term));
        // A type-specific list extractor provides stronger class evidence than
        // repeating the category noun in the source sentence.
        let answer_class_bound = plan.answer_class_terms.is_empty()
            || answer_class_overlap > 0
            || bounded_list_response_bridge
            || achievement_answer_class_bound
            || list_value_candidate;
        let nearby_dialogue_lower = nearby_dialogue_contexts
            .get(&(atom.memory_id, atom.context_group))
            .cloned()
            .unwrap_or_else(|| turn_context_lower.clone());
        let qualifier_context_lower = format!("{contextual_lower} {nearby_dialogue_lower}");
        let matched_qualifier_terms = plan
            .qualifier_terms
            .iter()
            .filter(|term| text_matches_lexical_term(&qualifier_context_lower, term))
            .collect::<Vec<_>>();
        // A named entity must occur on the fact itself. Wider context may
        // resolve an immediate pronoun, but cannot make a team sentence a
        // fact about a nearby player.
        let participant_relationship_context =
            participant_relationship_context_matches(plan, atom, &snapshot);
        let qualifier_terms_bound_here = if plan.participant_relationship_qualifier {
            participant_relationship_context
        } else {
            !matched_qualifier_terms.is_empty()
        };

        let direct_identity_lower = format!(
            "{} {} {}",
            atom.speaker.as_deref().unwrap_or_default().to_lowercase(),
            atom.entity.as_deref().unwrap_or_default().to_lowercase(),
            lower
        );
        let matched_named_entities = plan
            .named_entities
            .iter()
            .filter(|entity| {
                query_entity_matches(plan, &direct_identity_lower, entity)
                    || adjacent_question.is_some_and(|question| {
                        query_entity_matches(plan, &question.text.to_lowercase(), entity)
                    })
            })
            .collect::<Vec<_>>();
        let named_overlap = matched_named_entities.len();
        let named_bound = plan.named_entities.is_empty() || named_overlap > 0;
        let mut score = atom.base_score.clamp(-1.0, 1.0) * 0.25
            + query_weight * 0.24
            + action_weight * 0.32
            + object_weight * 0.28
            + qualifier_weight * 0.20;
        let media_provenance_bridge = media_provenance_response_bridge(
            plan,
            atom,
            &turn_context_lower,
            &snapshot,
            subject_requires_speaker,
        );
        let structured_profile_summary_bridge = plan_requests_structured_profile(plan)
            && atom
                .subject_binding
                .iter()
                .any(|binding| binding == "structured_profile_summary");
        let base_bridged_action_bound = action_bound
            || media_provenance_bridge
            || co_participant_event
            || structured_profile_summary_bridge;
        let base_bridged_object_bound =
            object_bound || media_provenance_bridge || structured_profile_summary_bridge;
        let bridged_named_bound = named_bound || media_provenance_bridge;
        // References in body text must not override an explicit containing section.
        let structural_selector_matches = if atom.section_scope.is_empty() {
            matching_structural_selectors(
                &format!("{} {}", atom.source.to_lowercase(), contextual_lower),
                &plan.structural_selectors,
            )
        } else {
            plan.structural_selectors
                .iter()
                .filter(|selector| atom.section_scope.iter().any(|present| structural_selector_contains(selector, present)))
                .cloned()
                .collect()
        };
        let structural_selector_bound =
            plan.structural_selectors.is_empty() || !structural_selector_matches.is_empty();
        let mut slots = BTreeSet::new();

        if media_provenance_bridge {
            score += 1.65;
            slots.insert("media_provenance_bridge".to_string());
            if !plan.action_terms.is_empty() {
                slots.insert("action".to_string());
            }
            if !plan.object_terms.is_empty() {
                slots.insert("object".to_string());
            }
            if !plan.named_entities.is_empty() {
                slots.insert("named_entity".to_string());
            }
            slots.insert("requested_modality".to_string());
        }
        if co_participant_event {
            score += 1.45;
            slots.insert("co_participant_event".to_string());
            if !plan.action_terms.is_empty() {
                slots.insert("action".to_string());
            }
        }
        if structured_profile_summary_bridge {
            slots.insert("structured_profile_summary".to_string());
            if !plan.action_terms.is_empty() {
                slots.insert("action".to_string());
            }
            if !plan.object_terms.is_empty() {
                slots.insert("object".to_string());
            }
        }

        let mut matched_people = 0usize;
        for person in &plan.people {
            let person_requires_speaker = subject_requires_speaker
                && plan.subject.as_ref().is_some_and(|subject| {
                    normalize_person_key(subject) == normalize_person_key(person)
                });
            if atom_matches_person(atom, &contextual_lower, person, person_requires_speaker) {
                matched_people += 1;
                score += 0.48;
                slots.insert(person_slot_name(person));
            }
        }
        let subject_bound = plan.subject.as_ref().is_none_or(|subject| {
            atom_matches_subject(atom, &contextual_lower, subject, subject_requires_speaker)
        });
        if subject_bound && plan.subject.is_some() {
            slots.insert("subject".to_string());
        }
        let list_unresolved_reference = plan.grounded_list_kind
            == Some(crate::memory_engine_summary::GroundedListKind::Books)
            && atom.kind.grounding_role() == GroundingRole::Original
            && list_values.is_empty()
            && subject_bound
            && crate::memory_engine_summary::source_has_unresolved_book_reference(&atom.text);
        if list_unresolved_reference {
            score += 0.30;
            slots.insert("list_unresolved_reference".to_string());
        }
        let list_value_bound = list_value_candidate
            && subject_bound
            && list_value_participant_scope_matches(
                plan,
                atom,
                &lower,
                &snapshot,
                adjacent_question,
            );
        let decompression_strength = list_value_bound
            .then(|| destress_relation_strength(plan, atom, &turn_context_lower))
            .unwrap_or_default();
        let bridged_action_bound = base_bridged_action_bound || list_value_bound;
        let bridged_object_bound = base_bridged_object_bound || list_value_bound;
        if list_value_bound {
            score += 1.35 + list_values.len().min(4) as f32 * 0.12;
            slots.insert("list_value".to_string());
            for value in &list_values {
                let key = normalize_for_dedup(value);
                if !key.is_empty() {
                    slots.insert(format!("list_value_key:{key}"));
                }
            }
            if same_turn_list_value_bridge {
                slots.insert("same_turn_list_value_bridge".to_string());
            }
            if decompression_strength > 0 {
                slots.insert(format!("decompression_strength:{decompression_strength}"));
            }
            if plan.list_requires_household_context {
                slots.insert("list_participant_scope".to_string());
            }
            if plan.household_dependent_preference {
                slots.insert("household_dependent_preference".to_string());
            }
        }

        if achievement_extremum_bound {
            score += 0.96;
            slots.insert("achievement_extremum".to_string());
        }
        if initiation_response_bridge && subject_bound {
            score += 0.70;
            slots.insert("initiation_response_context_bridge".to_string());
        }
        if initiation_event_bound && subject_bound && object_bound && named_bound {
            // Starting a practice, job, device workflow, or other activity
            // is a distinct answer type. A later opinion about the same topic
            // cannot stand in for the concrete initiating event.
            score += 1.30;
            slots.insert("initiation_event".to_string());
        }
        if bounded_list_response_bridge && subject_bound {
            score += 1.45;
            slots.insert("bounded_list_response_bridge".to_string());
        }
        if named_overlap > 0 {
            score += named_overlap.min(4) as f32 * 0.42;
            slots.insert("named_entity".to_string());
            slots.extend(
                matched_named_entities
                    .iter()
                    .map(|entity| named_entity_slot_name(entity)),
            );
        }
        if !structural_selector_matches.is_empty() {
            score += structural_selector_matches.len().min(2) as f32 * 1.05;
            slots.insert("structural_selector".to_string());
            for selector in &structural_selector_matches {
                slots.insert(structural_selector_slot_name(selector));
            }
        }

        if action_bound && !plan.action_terms.is_empty() {
            score += 0.68 + action_overlap.min(4) as f32 * 0.14;
            slots.insert("action".to_string());
        } else if !plan.action_terms.is_empty()
            && atom.kind.grounding_role() == GroundingRole::Original
        {
            score -= 0.72;
        }

        if action_object_complement_required && action_object_complement_bound {
            score += 0.42;
            slots.insert("action_object_complement".to_string());
        } else if action_object_complement_required
            && atom.kind.grounding_role() == GroundingRole::Original
        {
            score -= 0.95;
        }

        if observed_action_fact_required && observed_action_fact_bound {
            score += 0.46;
            slots.insert("observed_action_fact".to_string());
        } else if observed_action_fact_required
            && atom.kind.grounding_role() == GroundingRole::Original
        {
            score -= 1.10;
        }

        if plan.categorical_attribute_kind.is_some() && categorical_attribute_value_bound {
            score += 1.20;
            slots.insert("categorical_attribute_value".to_string());
        } else if plan.categorical_attribute_kind.is_some()
            && atom.kind.grounding_role() == GroundingRole::Original
        {
            // A related theme, acknowledgement, or nearby dialogue context is
            // not a categorical value for the requested subject.
            score -= 1.25;
        }

        if duration_value_required && duration_value_bound {
            score += 1.15;
            slots.insert("duration_value".to_string());
        } else if duration_value_required && atom.kind.grounding_role() == GroundingRole::Original {
            // A topic mention or acknowledgement cannot answer a duration
            // question unless this source sentence supplies a duration value.
            score -= 1.25;
        }

        if object_bound && !plan.object_terms.is_empty() {
            score += 0.62
                + object_overlap.min(4) as f32 * 0.16
                + exact_object_overlap.min(4) as f32 * 0.28;
            slots.insert("object".to_string());
        } else if !plan.object_terms.is_empty()
            && atom.kind.grounding_role() == GroundingRole::Original
        {
            score -= 0.68;
        }

        if direct_description_value {
            // For a description request, a statement that directly predicates a
            // property of the requested object is more answerable than a merely
            // related activity carried by the same topic context.
            score += 1.30;
            slots.insert("description_value".to_string());
        }

        if answer_class_bound && !plan.answer_class_terms.is_empty() {
            score += 0.72 + answer_class_overlap.min(3) as f32 * 0.14;
            slots.insert("answer_class".to_string());
        } else if !plan.answer_class_terms.is_empty()
            && atom.kind.grounding_role() == GroundingRole::Original
        {
            // A list about movies, reports, or another explicit result class
            // must not be filled by merely adjacent topical statements.
            score -= 0.92;
        }
        let source_qualifier_anchor = plan.subject.is_some()
            && atom.context_group != usize::MAX
            && atom.speaker.as_deref().is_some_and(|speaker| {
                let speaker = normalize_person_key(speaker);
                qualifier_anchors.get(&atom.memory_id).is_some_and(|anchors| {
                    anchors.iter().any(|(group, anchor_speaker)| {
                        atom.context_group.abs_diff(*group) > 4 && *anchor_speaker == speaker
                    })
                })
            });
        let source_qualifier_bridge = source_qualifier_anchor
            && action_bound
            && direct_action_bound
            && subject_bound
            && named_bound
            && atom.kind.grounding_role() == GroundingRole::Original
            && atom.context_group != usize::MAX;
        // A relational qualifier can be stored in a separate record. Permit
        // that bridge only for the same explicit action subject and only when
        // one supporting record covers every qualifier term.
        let cross_record_qualifier_bridge = !plan.qualifier_terms.is_empty()
            && source_level_qualifier_bridge_allowed(plan)
            && plan.subject_is_action_agent
            && action_bound
            && direct_action_bound
            && subject_bound
            && named_bound
            && atom.kind.grounding_role() == GroundingRole::Original
            && atom.speaker.as_deref().is_some_and(|speaker| {
                cross_record_qualifiers
                    .get(&normalize_person_key(speaker))
                    .is_some_and(|memories| memories.iter().any(|memory| *memory != atom.memory_id))
            });
        if qualifier_terms_bound_here
            && atom.kind.grounding_role() == GroundingRole::Original
            && ((!plan.participant_relationship_qualifier && subject_bound)
                || participant_relationship_context)
            && named_bound
        {
            score += 0.42 + matched_qualifier_terms.len().min(4) as f32 * 0.16;
            slots.insert("qualifier_context".to_string());
            for term in &matched_qualifier_terms {
                slots.insert(qualifier_slot_name(term));
            }
        } else if source_qualifier_bridge {
            score += 1.05;
            slots.insert("source_qualifier_bridge".to_string());
        } else if cross_record_qualifier_bridge {
            score += 0.70;
            slots.insert("cross_record_qualifier_bridge".to_string());
        }

        let qualifier_bound = plan.qualifier_terms.is_empty()
            || qualifier_terms_bound_here
            || source_qualifier_bridge
            || cross_record_qualifier_bridge;
        let collective_participant_event_bridge = !subject_bound
            && plan.subject.is_some()
            && plan
                .query_terms
                .iter()
                .any(|term| matches!(term.as_str(), "family" | "families"))
            && atom.kind.grounding_role() == GroundingRole::Original
            && action_bound
            && direct_action_bound
            && qualifier_terms_bound_here
            && lower
                .split_whitespace()
                .any(|term| matches!(term, "we" | "our"));
        if collective_participant_event_bridge {
            score += 0.52;
            slots.insert("collective_participant_context_bridge".to_string());
        }
        let own_query_overlap = matching_term_count(&lower, &plan.query_terms);
        let turn_query_overlap = matching_term_count(&turn_context_lower, &plan.query_terms);
        let adjacent_question_bound = adjacent_question.is_some() && query_overlap > 0;
        let primary_object_in_statement = plan.object_terms.is_empty()
            || matching_lexical_term_count(&lower, &plan.object_terms) > 0;
        let direct_complementary_action_anchor = plan.requires_complementary_action
            && statement_signals_complementary_action(&lower)
            && primary_object_in_statement;
        let complementary_action_bound = !plan.requires_complementary_action
            || direct_complementary_action_anchor
            || adjacent_question.is_some_and(|question| {
                statement_signals_complementary_action(&question.text.to_lowercase())
            });
        // A session-level date may support a direct event fact in the same
        // record. It cannot turn an acknowledgement or merely related
        // sentence into a dated event.
        let session_date_precision_available = session_dates.contains_key(&atom.memory_id)
            || (temporal_granularity_accepts_month_year(plan.temporal_granularity)
                && session_month_years.contains_key(&atom.memory_id));
        let direct_statement_session_date_bound =
            matches!(atom.kind, EvidenceKind::Statement | EvidenceKind::Event)
                && session_date_precision_available
                && subject_bound
                && direct_action_bound
                && primary_object_in_statement
                && !has_unresolved_temporal_reference(&lower)
                && !is_low_information_statement(&atom.text);
        // Context may disambiguate an event, but it cannot lend its date to a
        // different statement from the same dialogue turn. Native media is
        // the bounded exception: projection attaches it to the exact source
        // line that supplied its event text and caption.
        let native_media_same_turn_time_bound = atom.kind == EvidenceKind::NativeMedia
            && atom_has_event_time_support(
                plan,
                atom,
                &turn_context_lower,
                &session_dates,
                &session_month_years,
            );
        let direct_temporal_target_supported = plan.answer_target != AnswerTarget::Temporal
            || atom_has_event_time_support(
                plan,
                atom,
                &lower,
                &session_dates,
                &session_month_years,
            )
            || direct_statement_session_date_bound
            || native_media_same_turn_time_bound;
        let temporal_constraint_bound = atom_satisfies_temporal_constraint(
            &plan.temporal_constraint,
            atom,
            &contextual_lower,
            &session_dates,
        );
        let relative_time_reference_bound =
            relative_time_reference_matches(&plan.time_terms, &contextual_lower);
        let native_media_turn_bound =
            atom.kind == EvidenceKind::NativeMedia && turn_query_overlap > 0;
        let native_media_preferred = native_media_is_answer_preferred(plan);
        let relationship_actor_conflict = plan.participant_relationship_qualifier
            && !subject_bound
            && participant_relationship_context
            && atom.kind.grounding_role() == GroundingRole::Original
            && action_bound
            && direct_action_bound
            && object_bound
            && named_bound
            && answer_class_bound
            && complementary_action_bound
            && qualifier_bound
            && (plan.answer_target != AnswerTarget::Initiation || initiation_event_bound)
            && direct_temporal_target_supported
            && temporal_constraint_bound
            && relative_time_reference_bound
            && !is_low_information_statement(&atom.text)
            && question_statement_can_fill_answer_slot(plan, &atom.text);
        if relationship_actor_conflict {
            score += 0.35;
            slots.insert("relationship_actor_conflict".to_string());
        }

        let original_answer_atom = atom.kind.grounding_role() == GroundingRole::Original
            && bridged_action_bound
            && action_object_complement_bound
            && observed_action_fact_bound
            && categorical_attribute_value_bound
            && duration_value_bound
            && location_value_bound
            && (plan.grounded_list_kind.is_none() || list_value_bound)
            && structural_selector_bound
            && bridged_object_bound
            && (subject_bound
                || relationship_actor_conflict
                || collective_participant_event_bridge)
            && bridged_named_bound
            && answer_class_bound
            && complementary_action_bound
            && qualifier_bound
            && household_dependent_preference_bound
            && (!plan.achievement_extremum || achievement_extremum_bound)
            && (plan.answer_target != AnswerTarget::Initiation || initiation_event_bound)
            && direct_temporal_target_supported
            && temporal_constraint_bound
            && relative_time_reference_bound
            && (own_query_overlap > 0
                || adjacent_question_bound
                || initiation_response_bridge
                || native_media_turn_bound
                || media_provenance_bridge
                || co_participant_event
                || (plan.temporal_constraint.is_active() && temporal_constraint_bound)
                || plan.query_terms.is_empty()
                || (plan.categorical_attribute_kind.is_some()
                    && categorical_attribute_value_bound)
                || list_value_bound
                || structured_profile_summary_bridge
                || summary_label_bound
                || achievement_extremum_bound
                || semantic_event_scope_bound)
            && !is_low_information_statement(&atom.text)
            && question_statement_can_fill_answer_slot(plan, &atom.text);
        // Lexical action overlap is useful for precision, but it is not a
        // valid reason to throw away a fact that is otherwise bound to the
        // requested person, object/entity, answer class, and target type.
        // This is deliberately a weaker tier and still requires a direct
        // query anchor; it does not reopen generic fallback generation.
        let has_named_anchor = !plan.named_entities.is_empty() && named_bound;
        let has_specific_anchor = object_overlap > 0
            || has_named_anchor
            || (plan.categorical_attribute_kind.is_some() && categorical_attribute_value_bound)
            || list_value_bound
            || semantic_event_scope_bound;
        let constrained_backoff_fact = !original_answer_atom
            && atom.kind.grounding_role() == GroundingRole::Original
            && !action_bound
            && action_object_complement_bound
            && observed_action_fact_bound
            && categorical_attribute_value_bound
            && duration_value_bound
            && location_value_bound
            && (plan.grounded_list_kind.is_none() || list_value_bound)
            && subject_bound
            && (plan.people.is_empty() || matched_people == plan.people.len())
            && named_bound
            && structural_selector_bound
            && object_bound
            && answer_class_bound
            && qualifier_bound
            && complementary_action_bound
            && household_dependent_preference_bound
            && (!plan.achievement_extremum || achievement_extremum_bound)
            && has_specific_anchor
            && (plan.answer_target != AnswerTarget::Initiation || initiation_event_bound)
            && direct_temporal_target_supported
            && temporal_constraint_bound
            && relative_time_reference_bound
            && !is_low_information_statement(&atom.text)
            && question_statement_can_fill_answer_slot(plan, &atom.text);
        if original_answer_atom {
            score += 1.05;
            slots.insert("answer_bearing_fact".to_string());
        } else if constrained_backoff_fact {
            score += 0.18;
            slots.insert("action".to_string());
            slots.insert("answer_bearing_fact".to_string());
            slots.insert("constrained_backoff_fact".to_string());
        } else if atom.kind.grounding_role() == GroundingRole::Original {
            score -= 0.45;
            if !plan.people.is_empty() && matched_people == 0 {
                score -= 0.40;
            }
        }
        if (original_answer_atom || constrained_backoff_fact)
            && plan.temporal_constraint.is_active()
        {
            score += 1.35;
            slots.insert("temporal_constraint".to_string());
            slots.insert("temporal_scope".to_string());
        }
        if (original_answer_atom || constrained_backoff_fact)
            && requires_relative_time_reference(&plan.time_terms)
            && relative_time_reference_bound
        {
            score += 1.15;
            slots.insert("relative_time_reference".to_string());
            slots.insert("temporal_scope".to_string());
        }
        if (original_answer_atom || constrained_backoff_fact) && plan.requires_complementary_action
        {
            slots.insert("complementary_action".to_string());
            if direct_complementary_action_anchor {
                slots.insert("complementary_action_anchor".to_string());
            }
        }
        if (original_answer_atom || constrained_backoff_fact)
            && plan.answer_target == AnswerTarget::Location
            && location_value_bound
        {
            score += 0.90;
            slots.insert("location".to_string());
        }
        if native_media_turn_bound && native_media_preferred {
            score += 0.55;
        } else if atom.kind == EvidenceKind::NativeMedia
            && plan
                .requested_modalities
                .iter()
                .all(|modality| modality == "text")
        {
            score -= 0.55;
        }

        if plan.temporal_scope != "any" {
            if atom.kind == EvidenceKind::SessionDate
                || contains_date_like_value(&lower)
                || contains_bare_calendar_month_reference(&lower)
            {
                score += if original_answer_atom || constrained_backoff_fact {
                    0.42
                } else {
                    0.18
                };
                slots.insert("temporal_scope".to_string());
            }
            // A session date can constrain an event, but it cannot turn every
            // unrelated sentence in that record into a dated event. Event time
            // belongs to the original statement only when it carries an
            // explicit date or a resolvable/session-contemporaneous time.
            if (original_answer_atom || constrained_backoff_fact)
                && plan.answer_target == AnswerTarget::Temporal
                && direct_temporal_target_supported
            {
                score += 0.42;
                slots.insert("event_time".to_string());
            }
        } else if matches!(
            atom.kind,
            EvidenceKind::SessionDate | EvidenceKind::SystemMetadata
        ) {
            score -= 0.35;
        }

        if plan.cardinality == Cardinality::Count
            && (original_answer_atom || constrained_backoff_fact)
        {
            slots.insert("quantity".to_string());
        }
        if plan.cardinality == Cardinality::List
            && (original_answer_atom || constrained_backoff_fact)
        {
            slots.insert("list_scope".to_string());
        }
        if !plan.quantity_terms.is_empty()
            && plan
                .quantity_terms
                .iter()
                .any(|term| text_matches_term(&contextual_lower, term))
        {
            score += 0.24;
            slots.insert("quantity".to_string());
        }
        if plan.inference_type != InferenceType::Direct
            && (original_answer_atom || constrained_backoff_fact)
        {
            slots.insert("inference_basis".to_string());
        }
        if !plan.negation_terms.is_empty()
            && plan
                .negation_terms
                .iter()
                .any(|term| contextual_lower.contains(term))
        {
            score += 0.34;
            slots.insert("negation".to_string());
        }
        if !plan.condition_terms.is_empty()
            && strict_condition_terms_match_evidence(plan, &contextual_lower)
        {
            score += 0.28;
            slots.insert("condition".to_string());
        }

        let protected_overlap = matching_term_count(&contextual_lower, &plan.protected_terms);
        score += protected_overlap.min(6) as f32 * 0.10;
        if plan
            .requested_modalities
            .iter()
            .any(|modality| modality == &atom.modality)
        {
            score += 0.25;
            slots.insert("requested_modality".to_string());
        }

        score += match atom.kind.grounding_role() {
            GroundingRole::Original => 0.16,
            GroundingRole::Context => 0.0,
            GroundingRole::ExpansionOnly => -0.55,
        };
        atom.score = score;
        atom.supports_slots = slots.into_iter().collect();
    }
    if plan.answer_shape == AnswerShape::ActionSet {
        promote_strategy_context_bridges(plan, atoms);
    }
    promote_source_qualifier_contexts(plan, atoms);
    if plan.answer_shape == AnswerShape::CountAggregate {
        promote_count_event_bridges(plan, atoms, &session_dates);
    }
    if plan.recommendation_requested {
        promote_preference_recommendation_bridges(plan, atoms);
    }
    retain_multi_entity_inference_contexts(plan, atoms);
}

fn query_entity_matches(plan: &QueryPlan, text_lower: &str, entity: &str) -> bool {
    if is_multi_entity_inference(plan) {
        // An unverified first/last-word alias cannot connect different entities.
        let entity_lower = entity.to_lowercase();
        text_lower
            .match_indices(&entity_lower)
            .any(|(start, matched)| {
                let end = start + matched.len();
                text_lower[..start]
                    .chars()
                    .next_back()
                    .is_none_or(|c| !c.is_alphanumeric())
                    && text_lower[end..]
                        .chars()
                        .next()
                        .is_none_or(|c| !c.is_alphanumeric())
            })
    } else {
        text_matches_named_entity(text_lower, entity)
    }
}

fn retain_multi_entity_inference_contexts(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    if !is_multi_entity_inference(plan) {
        return;
    }
    let Some(subject) = plan.subject.as_deref() else {
        return;
    };
    for atom in atoms {
        if atom.kind.grounding_role() != GroundingRole::Original {
            continue;
        }
        let entity_slots = plan
            .named_entities
            .iter()
            .map(|entity| named_entity_slot_name(entity));
        let complete_entity_match = entity_slots
            .clone()
            .all(|slot| atom.supports_slots.contains(&slot));
        if atom_is_answer_bearing(atom) && complete_entity_match {
            continue;
        }
        // Retain a premise as a premise. It cannot cover the requested outcome.
        atom.supports_slots
            .retain(|slot| slot != "answer_bearing_fact" && slot != "inference_basis");
        let direct_entities = plan
            .named_entities
            .iter()
            .filter(|entity| query_entity_matches(plan, &atom.text.to_lowercase(), entity))
            .map(|entity| named_entity_slot_name(entity))
            .collect::<Vec<_>>();
        let subject_bound = atom_matches_subject(atom, &atom.text.to_lowercase(), subject, true);
        if direct_entities.is_empty()
            || !subject_bound
            || is_low_information_statement(&atom.text)
            || atom.text.trim_end().ends_with(['?', '？'])
        {
            continue;
        }
        atom.supports_slots
            .retain(|slot| !slot.starts_with("named_entity:"));
        atom.supports_slots.extend(direct_entities);
        atom.supports_slots
            .push("inference_entity_context".to_string());
        atom.supports_slots.push("subject".to_string());
        atom.supports_slots.push(person_slot_name(subject));
        atom.supports_slots.sort();
        atom.supports_slots.dedup();
    }
}

fn recommendation_capability_label(value: &str) -> Option<&'static str> {
    let lower = value.to_lowercase();
    if contains_any(
        &lower,
        &["writing", "write ", "wrote ", "article", "blog", "journal"],
    ) {
        Some("writing")
    } else if contains_any(&lower, &["photography", "photograph", "photo shoot"]) {
        Some("photography")
    } else if contains_any(&lower, &["drawing", "draw ", "sketch", "illustrat"]) {
        Some("drawing")
    } else if contains_any(&lower, &["painting", "paint "]) {
        Some("painting")
    } else if contains_any(&lower, &["coding", "code ", "programming", "developing"]) {
        Some("coding")
    } else if contains_any(&lower, &["cooking", "cook ", "baking", "bake "]) {
        Some("cooking")
    } else if contains_any(&lower, &["designing", "design "]) {
        Some("design")
    } else if contains_any(&lower, &["gardening", "garden "]) {
        Some("gardening")
    } else if contains_any(
        &lower,
        &["music", "playing guitar", "playing piano", "singing"],
    ) {
        Some("music")
    } else {
        None
    }
}

fn promote_preference_recommendation_bridges(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    let Some(subject) = plan.subject.as_deref() else {
        return;
    };
    let Some(topic) = plan.recommendation_topic_terms.first() else {
        return;
    };
    let subject_matches = |atom: &EvidenceAtom| {
        atom.kind.grounding_role() == GroundingRole::Original
            && atom.speaker.as_deref().is_some_and(|speaker| {
                normalize_person_key(speaker) == normalize_person_key(subject)
            })
    };
    let capability = atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| {
            subject_matches(atom) && recommendation_capability_label(&atom.text).is_some()
        })
        .max_by(|(_, left), (_, right)| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(index, _)| index);
    let topic_evidence = atoms
        .iter()
        .enumerate()
        .filter(|(_, atom)| {
            subject_matches(atom) && text_matches_lexical_term(&atom.text.to_lowercase(), topic)
        })
        .max_by(|(_, left), (_, right)| {
            left.score
                .partial_cmp(&right.score)
                .unwrap_or(Ordering::Equal)
        })
        .map(|(index, _)| index);
    let (Some(capability), Some(topic_evidence)) = (capability, topic_evidence) else {
        return;
    };
    for (index, slot, floor) in [
        (capability, "recommendation_capability", 6.5_f32),
        (topic_evidence, "recommendation_topic", 6.4_f32),
    ] {
        let atom = &mut atoms[index];
        let mut slots = atom.supports_slots.iter().cloned().collect::<BTreeSet<_>>();
        slots.insert("answer_bearing_fact".to_string());
        slots.insert("inference_basis".to_string());
        slots.insert("subject".to_string());
        slots.insert(person_slot_name(subject));
        slots.insert(slot.to_string());
        atom.score = atom.score.max(floor);
        atom.supports_slots = slots.into_iter().collect();
    }
}

fn promote_count_event_bridges(
    plan: &QueryPlan,
    atoms: &mut [EvidenceAtom],
    session_dates: &HashMap<u64, chrono::NaiveDate>,
) {
    let Some(subject) = plan.subject.as_deref() else {
        return;
    };
    if plan.object_terms.is_empty() {
        return;
    }
    let anchors = atoms
        .iter()
        .filter_map(|atom| {
            let speaker = atom.speaker.as_deref()?;
            (atom_is_answer_bearing(atom)
                && atom.supports_slots.iter().any(|slot| slot == "action")
                && atom.supports_slots.iter().any(|slot| slot == "object")
                && normalize_person_key(speaker) == normalize_person_key(subject))
            .then_some((
                atom.atom_id.clone(),
                atom.memory_id,
                speaker.to_string(),
                atom.score,
            ))
        })
        .collect::<Vec<_>>();
    if anchors.is_empty() {
        return;
    }

    // A contemporaneous phrase such as "this injury" can identify a new
    // event in another session, but only after a separate source has already
    // bound the requested object and action. Keep one bridge per source so a
    // single conversation cannot inflate an event count.
    let mut bridged_memory_ids = HashSet::new();
    for candidate in atoms.iter_mut() {
        if candidate.kind.grounding_role() != GroundingRole::Original
            || !matches!(
                candidate.kind,
                EvidenceKind::Statement | EvidenceKind::Event
            )
            || atom_is_answer_bearing(candidate)
            || candidate.supports_slots.iter().any(|slot| slot == "object")
            || !candidate.supports_slots.iter().any(|slot| slot == "action")
            || !candidate
                .supports_slots
                .iter()
                .any(|slot| slot == "subject")
            || !statement_has_count_event_anaphora(plan, &candidate.text)
        {
            continue;
        }
        let Some(candidate_speaker) = candidate.speaker.as_deref() else {
            continue;
        };
        let Some(session_date) = session_dates.get(&candidate.memory_id) else {
            continue;
        };
        if crate::memory_engine_summary::relative_event_date_answer(&candidate.text, *session_date)
            .is_none()
        {
            continue;
        }
        let Some((anchor_atom_id, anchor_score)) = anchors
            .iter()
            .filter(|(_, memory_id, speaker, _)| {
                *memory_id != candidate.memory_id
                    && normalize_person_key(speaker) == normalize_person_key(candidate_speaker)
            })
            .map(|(atom_id, _, _, score)| (atom_id.clone(), *score))
            .max_by(|left, right| left.1.partial_cmp(&right.1).unwrap_or(Ordering::Equal))
        else {
            continue;
        };
        if !bridged_memory_ids.insert(candidate.memory_id) {
            continue;
        }

        let mut slots = candidate
            .supports_slots
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        slots.insert("object".to_string());
        slots.insert("answer_bearing_fact".to_string());
        slots.insert("quantity".to_string());
        slots.insert("count_event_bridge".to_string());
        slots.insert(format!("count_event_bridge_anchor:{anchor_atom_id}"));
        candidate.score = candidate.score.max(anchor_score - 1.25);
        candidate.supports_slots = slots.into_iter().collect();
    }
}

fn statement_has_count_event_anaphora(plan: &QueryPlan, value: &str) -> bool {
    let lower = value.to_lowercase();
    plan.action_terms
        .iter()
        .chain(plan.object_terms.iter())
        .flat_map(|term| term_forms(term))
        .filter(|term| term.chars().count() >= 3)
        .any(|term| {
            lower.contains(&format!("this {term}"))
                || lower.contains(&format!("{term}'s"))
                || lower.contains(&format!("{term} is"))
                || lower.contains(&format!("{term} has"))
        })
}

fn promote_strategy_context_bridges(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    let mut term_document_ids = HashMap::<String, BTreeSet<u64>>::new();
    let source_memory_ids = atoms
        .iter()
        .map(|atom| atom.memory_id)
        .collect::<BTreeSet<_>>();
    let maximum_bridge_document_frequency = ((source_memory_ids.len() + 2) / 3).max(2);
    for atom in atoms.iter().filter(|atom| {
        atom.kind.grounding_role() == GroundingRole::Original
            && matches!(atom.kind, EvidenceKind::Statement | EvidenceKind::Event)
    }) {
        for term in strategy_bridge_terms_for_plan(plan, &atom.text) {
            term_document_ids
                .entry(term)
                .or_default()
                .insert(atom.memory_id);
        }
    }

    let anchors = atoms
        .iter()
        .filter_map(|atom| {
            if !atom_is_answer_bearing(atom)
                || !atom.supports_slots.iter().any(|slot| slot == "action")
                || is_low_information_statement(&atom.text)
            {
                return None;
            }
            if !atom
                .supports_slots
                .iter()
                .any(|slot| slot == "complementary_action_anchor")
            {
                return None;
            }
            let speaker = atom.speaker.clone()?;
            let terms = strategy_bridge_terms_for_plan(plan, &atom.text)
                .into_iter()
                .filter(|term| {
                    term_document_ids.get(term).is_some_and(|documents| {
                        documents.len() <= maximum_bridge_document_frequency
                    })
                })
                .collect::<BTreeSet<_>>();
            (!terms.is_empty()).then_some((atom.memory_id, speaker, atom.score, terms))
        })
        .collect::<Vec<_>>();
    if anchors.is_empty() {
        return;
    }

    for candidate in atoms.iter_mut() {
        if !matches!(
            candidate.kind,
            EvidenceKind::Statement | EvidenceKind::Event
        ) || candidate.kind.grounding_role() != GroundingRole::Original
            || atom_is_answer_bearing(candidate)
            || is_low_information_statement(&candidate.text)
            || !strategy_statement_has_concrete_action(
                &candidate.text,
                candidate.speaker.as_deref(),
            )
        {
            continue;
        }
        let Some(candidate_speaker) = candidate.speaker.as_deref() else {
            continue;
        };
        let candidate_terms = strategy_bridge_terms_for_plan(plan, &candidate.text);
        if candidate_terms.is_empty() {
            continue;
        }
        let Some(anchor_score) = anchors
            .iter()
            .filter(|(memory_id, speaker, _, terms)| {
                *memory_id != candidate.memory_id
                    && normalize_person_key(speaker) == normalize_person_key(candidate_speaker)
                    && terms.iter().any(|term| candidate_terms.contains(term))
            })
            .map(|(_, _, score, _)| *score)
            .max_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal))
        else {
            continue;
        };

        let mut slots = candidate
            .supports_slots
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        slots.insert("action".to_string());
        slots.insert("answer_bearing_fact".to_string());
        slots.insert("strategy_context_bridge".to_string());
        if plan.requires_complementary_action {
            slots.insert("complementary_action".to_string());
        }
        if plan.subject.as_ref().is_some_and(|subject| {
            normalize_person_key(subject) == normalize_person_key(candidate_speaker)
        }) {
            slots.insert("subject".to_string());
        }
        for person in &plan.people {
            if normalize_person_key(person) == normalize_person_key(candidate_speaker) {
                slots.insert(person_slot_name(person));
            }
        }
        if plan
            .requested_modalities
            .iter()
            .any(|modality| modality == &candidate.modality)
        {
            slots.insert("requested_modality".to_string());
        }
        // Keep the bridge weaker than the direct anchor. It can supply an
        // additional action, but it cannot replace the fact that binds the
        // strategy to the user's stated goal.
        candidate.score = candidate.score.max(anchor_score - 2.25);
        candidate.supports_slots = slots.into_iter().collect();
    }
}

fn strategy_bridge_terms_for_plan(plan: &QueryPlan, value: &str) -> BTreeSet<String> {
    strategy_bridge_terms(value)
        .into_iter()
        .filter(|term| {
            !plan
                .object_terms
                .iter()
                .chain(plan.action_terms.iter())
                .chain(plan.qualifier_terms.iter())
                .any(|primary_term| {
                    lexical_term_forms(primary_term)
                        .iter()
                        .any(|form| form == term)
                })
        })
        .collect()
}

fn strategy_bridge_terms(value: &str) -> BTreeSet<String> {
    informative_terms(value)
        .into_iter()
        .filter(|term| term.chars().count() >= 4 || term.chars().any(is_cjk))
        .filter(|term| {
            !matches!(
                term.as_str(),
                "about"
                    | "also"
                    | "been"
                    | "could"
                    | "from"
                    | "great"
                    | "have"
                    | "here"
                    | "just"
                    | "little"
                    | "more"
                    | "really"
                    | "some"
                    | "that"
                    | "their"
                    | "there"
                    | "these"
                    | "they"
                    | "this"
                    | "very"
                    | "with"
                    | "would"
                    | "your"
            ) && !is_non_discriminative_strategy_bridge_term(term)
        })
        .collect()
}

fn is_non_discriminative_strategy_bridge_term(term: &str) -> bool {
    matches!(
        term,
        "activity"
            | "game"
            | "games"
            | "player"
            | "players"
            | "practice"
            | "sport"
            | "sports"
            | "stay"
            | "team"
            | "teams"
            | "train"
            | "training"
            | "workout"
            | "workouts"
            | "锻炼"
            | "训练"
            | "运动"
            | "球队"
            | "队伍"
    )
}

fn strategy_statement_has_concrete_action(value: &str, speaker: Option<&str>) -> bool {
    if value.contains('?') || value.contains('？') {
        return false;
    }
    let lower = value.to_lowercase();
    let explicit_actor = speaker
        .is_some_and(|speaker| text_matches_term(&lower, &speaker.to_lowercase()))
        || lower.starts_with("i ")
        || lower.starts_with("i'm")
        || lower.starts_with("i am")
        || lower.starts_with("i've")
        || lower.starts_with("we ")
        || lower.starts_with("we're")
        || lower.starts_with("we are")
        || lower.contains("我")
        || lower.contains("我们");
    if !explicit_actor {
        return false;
    }

    // A photo, a result, or a feeling can mention a workout without stating
    // an action the subject actually chose. Cross-record bridges must carry a
    // reproducible action, not merely topical context.
    let has_specific_action = contains_any(
        &lower,
        &[
            "try",
            "use",
            "practice",
            "train",
            "exercise",
            "work out",
            "rest",
            "schedule",
            "do yoga",
            "do pilates",
            "尝试",
            "使用",
            "练习",
            "训练",
            "锻炼",
            "休息",
            "安排",
        ],
    );
    if !has_specific_action {
        return false;
    }
    !contains_any(
        &lower,
        &[
            "here's a pic",
            "here is a pic",
            "photo of",
            "image of",
            "picture of",
            "照片",
            "图片",
        ],
    )
}

pub(crate) fn rerank_answer_candidates(query: &str, items: &[RetrievedItem]) -> Vec<RetrievedItem> {
    let plan = build_query_plan(query, items);
    let required_slot_count = plan.required_slots.len().max(1);
    let mut all_atoms = items
        .iter()
        .flat_map(|item| {
            project_source(ProjectionInput {
                memory_id: item.id,
                content: &item.content,
                source: &item.source,
                entity: item.entity.as_deref(),
                score: item.score,
                is_event: item.event_metadata.is_some()
                    || item.content.trim_start().starts_with("event:"),
            })
        })
        .collect::<Vec<_>>();
    rank_evidence_atoms(&plan, &mut all_atoms);

    let mut reranked = items
        .iter()
        .cloned()
        .map(|mut item| {
            let item_atoms = all_atoms
                .iter()
                .filter(|atom| atom.memory_id == item.id)
                .collect::<Vec<_>>();
            let supported_slots = item_atoms
                .iter()
                .flat_map(|atom| atom.supports_slots.iter().cloned())
                .collect::<BTreeSet<_>>();
            let covered = plan
                .required_slots
                .iter()
                .filter(|slot| supported_slots.contains(*slot))
                .count();
            let best_original = item_atoms
                .iter()
                .filter(|atom| atom.kind.grounding_role() == GroundingRole::Original)
                .map(|atom| atom.score)
                .max_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal))
                .unwrap_or(item.score - 1.0);
            let answer_bearing = supported_slots.contains("answer_bearing_fact");
            let slot_ratio = covered as f32 / required_slot_count as f32;
            item.score =
                best_original + slot_ratio * 1.10 + if answer_bearing { 1.10 } else { -1.10 };
            item
        })
        .collect::<Vec<_>>();
    reranked.sort_by(|left, right| {
        right
            .score
            .partial_cmp(&left.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.id.cmp(&right.id))
    });
    reranked
}

#[allow(dead_code)]
fn rank_evidence_atoms_legacy(plan: &QueryPlan, atoms: &mut [EvidenceAtom]) {
    for atom in atoms {
        let lower = atom.text.to_lowercase();
        let overlap = plan
            .query_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        let action_overlap = plan
            .action_terms
            .iter()
            .filter(|term| lower.contains(term.as_str()))
            .count();
        let relevance_overlap = if plan.action_terms.is_empty() {
            overlap
        } else {
            action_overlap
        };
        let mut score = atom.base_score + (relevance_overlap.min(6) as f32 * 0.12);
        let mut slots = BTreeSet::new();
        if relevance_overlap > 0
            && matches!(
                atom.kind,
                EvidenceKind::Statement | EvidenceKind::Event | EvidenceKind::NativeMedia
            )
        {
            slots.insert("answer_bearing_fact".to_string());
        }
        if let Some(subject) = plan.subject.as_ref() {
            let subject_lower = subject.to_lowercase();
            let subject_match = atom
                .speaker
                .as_ref()
                .is_some_and(|value| value.to_lowercase().contains(&subject_lower))
                || atom
                    .entity
                    .as_ref()
                    .is_some_and(|value| value.to_lowercase() == subject_lower)
                || (atom.speaker.is_none() && lower.contains(&subject_lower));
            if subject_match {
                score += 0.45;
                slots.insert("subject".to_string());
            }
        }
        if plan.temporal_scope != "any" {
            if atom.kind == EvidenceKind::SessionDate || contains_date_like_value(&lower) {
                score += 0.3;
                slots.insert("temporal_scope".to_string());
            }
        } else if matches!(
            atom.kind,
            EvidenceKind::SessionDate | EvidenceKind::SystemMetadata
        ) {
            score -= 0.35;
        }
        if action_overlap > 0 {
            slots.insert("action".to_string());
        }
        if plan.cardinality != Cardinality::Single
            && matches!(atom.kind, EvidenceKind::Statement | EvidenceKind::Event)
            && relevance_overlap > 0
        {
            slots.insert("quantity_or_list_scope".to_string());
        }
        if plan.inference_type != InferenceType::Direct
            && matches!(atom.kind, EvidenceKind::Statement | EvidenceKind::Event)
            && relevance_overlap > 0
        {
            slots.insert("inference_basis".to_string());
        }
        if plan
            .requested_modalities
            .iter()
            .any(|modality| modality == &atom.modality)
        {
            score += 0.25;
            slots.insert("requested_modality".to_string());
        }
        if atom.kind == EvidenceKind::Entity {
            score -= 0.15;
        }
        atom.score = score;
        atom.supports_slots = slots.into_iter().collect();
    }
}

fn open_list_has_explicit_closure(answer_atoms: &[&EvidenceAtom]) -> bool {
    answer_atoms.iter().any(|atom| {
        let lower = atom.text.to_lowercase();
        contains_any(
            &lower,
            &[
                "only ",
                "just ",
                "that's all",
                "that is all",
                "those are all",
                "these are all",
                "all of my",
                "complete list",
                "只有",
                "仅",
                "就这",
                "全部",
                "所有",
            ],
        )
    })
}

fn is_referential_list_continuation(value: &str) -> bool {
    let lower = value.trim().to_lowercase();
    [
        "it ", "it's", "it is", "they ", "they're", "they are", "this ", "that ", "these ",
        "those ", "它", "这", "那", "这些", "那些",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
}

fn assess_slot_coverage(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    eligible_count: usize,
    candidate_pool_count: usize,
) -> Vec<AnswerSlotCoverageReceipt> {
    let answer_atoms = atoms
        .iter()
        .filter(|atom| {
            atom.kind.grounding_role() == GroundingRole::Original
                && atom
                    .supports_slots
                    .iter()
                    .any(|slot| slot == "answer_bearing_fact")
        })
        .collect::<Vec<_>>();
    let answer_ids = answer_atoms
        .iter()
        .map(|atom| atom.memory_id)
        .collect::<BTreeSet<_>>();
    // Bridge atoms are proof-only: they can cover a subject, action, location,
    // or qualifier needed by the final Claim without producing another output
    // node. The answer-bearing atom remains separately identifiable above.
    let proof_atoms = atoms
        .iter()
        .filter(|atom| {
            atom.kind.grounding_role() == GroundingRole::Original
                && (atom_is_answer_bearing(atom)
                    || atom
                        .supports_slots
                        .iter()
                        .any(|slot| slot == "evidence_chain_support"))
        })
        .collect::<Vec<_>>();
    let scope_complete = eligible_count > 0 && candidate_pool_count >= eligible_count;

    plan.required_slots
        .iter()
        .map(|slot| {
            let mut evidence_ids = match slot.as_str() {
                "answer_bearing_fact" => answer_ids.iter().copied().collect::<Vec<_>>(),
                "temporal_scope" => atoms
                    .iter()
                    .filter(|atom| {
                        atom.supports_slots
                            .iter()
                            .any(|value| value == "temporal_scope")
                            && answer_ids.contains(&atom.memory_id)
                    })
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                "event_time" => proof_atoms
                    .iter()
                    .filter(|atom| {
                        atom.supports_slots
                            .iter()
                            .any(|value| value == "event_time")
                    })
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                "subject"
                | "action"
                | "achievement_extremum"
                | "complementary_action"
                | "negation"
                | "condition"
                | "requested_modality" => proof_atoms
                    .iter()
                    .filter(|atom| atom.supports_slots.iter().any(|value| value == slot))
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                value if value.starts_with("named_entity:") => atoms
                    .iter()
                    .filter(|atom| {
                        atom_is_answer_bearing(atom) || atom_is_inference_entity_context(atom)
                    })
                    .filter(|atom| atom.supports_slots.iter().any(|slot| slot == value))
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                value if value.starts_with("person:") => proof_atoms
                    .iter()
                    .filter(|atom| atom.supports_slots.iter().any(|slot| slot == value))
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                "location" => proof_atoms
                    .iter()
                    .filter(|atom| atom.supports_slots.iter().any(|value| value == "location"))
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                "quantity" | "list_scope" | "answer_class" | "inference_basis" => proof_atoms
                    .iter()
                    .filter(|atom| atom.supports_slots.iter().any(|value| value == slot))
                    .map(|atom| atom.memory_id)
                    .collect::<Vec<_>>(),
                _ => evidence_ids_for_slot(atoms, slot),
            };
            evidence_ids.sort_unstable();
            evidence_ids.dedup();

            let status = if evidence_ids.is_empty() {
                "missing"
            } else if slot == "action" || slot == "complementary_action" {
                let all_action_support_is_bridged = answer_atoms
                    .iter()
                    .filter(|atom| atom.supports_slots.iter().any(|value| value == slot))
                    .all(|atom| {
                        atom.supports_slots.iter().any(|value| {
                            value == "constrained_backoff_fact"
                                || value == "strategy_context_bridge"
                        })
                    });
                if all_action_support_is_bridged {
                    "partial"
                } else {
                    "covered"
                }
            } else if slot == "temporal_scope" {
                let has_bound_context = evidence_ids.iter().any(|memory_id| {
                    let has_event = answer_ids.contains(memory_id);
                    let has_time = atoms.iter().any(|atom| {
                        atom.memory_id == *memory_id
                            && atom
                                .supports_slots
                                .iter()
                                .any(|value| value == "temporal_scope")
                    });
                    has_event && has_time
                });
                if has_bound_context {
                    "covered"
                } else {
                    "partial"
                }
            } else if slot.strip_prefix("structural_selector:")
                .is_some_and(|selector| !structural_range_fully_observed(selector, &answer_atoms)) {
                "partial"
            } else if slot == "event_time" {
                "covered"
            } else if slot == "qualifier_context" {
                let all_qualifiers_bound = plan.qualifier_terms.iter().all(|term| {
                    let term_slot = qualifier_slot_name(term);
                    atoms
                        .iter()
                        .any(|atom| atom.supports_slots.contains(&term_slot))
                });
                if all_qualifiers_bound {
                    "covered"
                } else {
                    "partial"
                }
            } else if slot == "inference_basis" {
                if evidence_ids.len() >= 2 {
                    "covered"
                } else {
                    "partial"
                }
            } else if slot == "quantity" && plan.cardinality == Cardinality::Count {
                if scope_complete {
                    "covered"
                } else {
                    "partial"
                }
            } else if slot == "list_scope" {
                match plan.list_scope.as_str() {
                    "exhaustive" if !scope_complete => "partial",
                    "open" if !scope_complete || !open_list_has_explicit_closure(&answer_atoms) => {
                        "partial"
                    }
                    "bounded" => {
                        let requested = requested_bounded_list_value_count(plan).unwrap_or(1);
                        let enumerated_values = answer_atoms
                            .iter()
                            .map(|atom| coordinated_proper_value_count(&atom.text))
                            .sum::<usize>();
                        if (requested == 1 && !evidence_ids.is_empty())
                            || enumerated_values >= requested
                        {
                            "covered"
                        } else {
                            "partial"
                        }
                    }
                    _ => "covered",
                }
            } else {
                "covered"
            };

            AnswerSlotCoverageReceipt {
                slot: slot.clone(),
                status: status.to_string(),
                evidence_ids,
            }
        })
        .collect()
}

#[allow(dead_code)]
fn assess_slot_coverage_legacy(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
) -> Vec<AnswerSlotCoverageReceipt> {
    let answer_ids = evidence_ids_for_slot(atoms, "answer_bearing_fact");
    plan.required_slots
        .iter()
        .map(|slot| {
            let mut evidence_ids = evidence_ids_for_slot(atoms, slot);
            let status = if slot == "temporal_scope" || slot == "subject" {
                let linked = evidence_ids
                    .iter()
                    .copied()
                    .filter(|memory_id| answer_ids.contains(memory_id))
                    .collect::<Vec<_>>();
                if !linked.is_empty() {
                    evidence_ids = linked;
                    "covered"
                } else if !evidence_ids.is_empty() && !answer_ids.is_empty() {
                    "partial"
                } else {
                    "missing"
                }
            } else if slot == "inference_basis" {
                if evidence_ids.len() >= 2 {
                    "covered"
                } else if evidence_ids.len() == 1 {
                    "partial"
                } else {
                    "missing"
                }
            } else if evidence_ids.is_empty() {
                "missing"
            } else {
                "covered"
            };
            evidence_ids.sort_unstable();
            evidence_ids.dedup();
            AnswerSlotCoverageReceipt {
                slot: slot.clone(),
                status: status.to_string(),
                evidence_ids,
            }
        })
        .collect()
}

fn answer_atom_citations(
    plan: &QueryPlan,
    answer_atom: &EvidenceAtom,
    atoms: &[EvidenceAtom],
) -> String {
    let mut citations = BTreeSet::new();
    citations.insert((answer_atom.memory_id, answer_atom.source.clone()));
    if !plan.qualifier_terms.is_empty() {
        for context in atoms.iter().filter(|atom| {
            atom.supports_slots
                .iter()
                .any(|slot| slot == "qualifier_context")
        }) {
            citations.insert((context.memory_id, context.source.clone()));
        }
    }
    // A concise answer can have one displayed node while depending on several
    // bounded evidence atoms. Cite the explicit chain rather than silently
    // dropping its supporting facts because the output is short.
    if answer_atom
        .supports_slots
        .iter()
        .any(|slot| slot == "evidence_chain_primary")
    {
        let relation_keys = answer_atom
            .supports_slots
            .iter()
            .filter_map(|slot| slot.strip_prefix("evidence_chain_relation:"))
            .collect::<HashSet<_>>();
        for companion in atoms.iter().filter(|candidate| {
            candidate
                .supports_slots
                .iter()
                .any(|slot| slot == "evidence_chain_member")
                && candidate.supports_slots.iter().any(|slot| {
                    slot.strip_prefix("evidence_chain_relation:")
                        .is_some_and(|relation| relation_keys.contains(relation))
                })
        }) {
            citations.insert((companion.memory_id, companion.source.clone()));
        }
    }
    citations
        .into_iter()
        .map(|(memory_id, source)| format!("[memory#{memory_id} | {source}]"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn build_session_anchored_temporal_answer(
    query: &str,
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
) -> Option<UnifiedAnswerDraft> {
    if plan.answer_target != AnswerTarget::Temporal {
        return None;
    }
    let requested_years = plan
        .time_terms
        .iter()
        .filter_map(|term| {
            (term.len() == 4 && term.chars().all(|character| character.is_ascii_digit()))
                .then(|| term.parse::<i32>().ok())
                .flatten()
        })
        .collect::<HashSet<_>>();
    for atom in atoms.iter().filter(|atom| {
        matches!(atom.kind, EvidenceKind::Statement | EvidenceKind::Event)
            && atom.kind.grounding_role() == GroundingRole::Original
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact")
    }) {
        let Some(session_date_atom) = atoms.iter().find(|candidate| {
            candidate.memory_id == atom.memory_id && candidate.kind == EvidenceKind::SessionDate
        }) else {
            continue;
        };
        let citations = answer_atom_citations(plan, atom, atoms);
        let chinese = query
            .chars()
            .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
        if let Some(session_date) =
            crate::memory_engine_summary::source_session_date(&session_date_atom.text)
        {
            let Some((answer_en, answer_zh, marker)) =
                crate::memory_engine_summary::relative_event_date_answer(&atom.text, session_date)
            else {
                continue;
            };
            let Some(year) = session_date.format("%Y").to_string().parse::<i32>().ok() else {
                continue;
            };
            if !requested_years.is_empty() && !requested_years.contains(&year) {
                continue;
            }
            let text = if chinese {
                format!("直接回答：日期是 {answer_zh}。{citations}")
            } else {
                format!("Direct answer: the date was {answer_en}. {citations}")
            };
            let claim_text = if chinese {
                format!("日期是 {answer_zh}。")
            } else {
                format!("The date was {answer_en}.")
            };
            let (evidence_atom_ids, evidence_source_anchors, mut derivation) =
                claim_evidence_binding_fields(&[atom, session_date_atom]);
            derivation.push("same_memory_session_date".to_string());
            derivation.push(format!("relative_event_date:{marker}"));
            derivation.sort();
            derivation.dedup();
            let evidence_ids = selected_memory_ids_for_atoms(&evidence_atom_ids, atoms);
            return Some(UnifiedAnswerDraft {
                text,
                draft_mode: "synthesized",
                rendered_claims: vec![AnswerClaimPlanReceipt {
                    claim_id: "claim-1".to_string(),
                    claim_type: "fact".to_string(),
                    text: Some(claim_text),
                    required_slots: plan.required_slots.clone(),
                    evidence_ids,
                    evidence_atom_ids,
                    evidence_source_anchors,
                    derivation,
                    support_status: "planned".to_string(),
                }],
            });
        }

        let Some(month_year) = source_session_month_year(&session_date_atom.text) else {
            continue;
        };
        if !temporal_granularity_accepts_month_year(plan.temporal_granularity)
            || has_unresolved_temporal_reference(&atom.text)
        {
            continue;
        }
        let year = number_regex()
            .find_iter(&month_year)
            .find_map(|value| value.as_str().parse::<i32>().ok());
        if !requested_years.is_empty() && !year.is_some_and(|year| requested_years.contains(&year))
        {
            continue;
        }
        let text = if chinese {
            format!("直接回答：日期是 {month_year}。{citations}")
        } else {
            format!("Direct answer: the date was {month_year}. {citations}")
        };
        let claim_text = if chinese {
            format!("日期是 {month_year}。")
        } else {
            format!("The date was {month_year}.")
        };
        let (evidence_atom_ids, evidence_source_anchors, mut derivation) =
            claim_evidence_binding_fields(&[atom, session_date_atom]);
        derivation.push("same_memory_session_month_year".to_string());
        derivation.sort();
        derivation.dedup();
        let evidence_ids = selected_memory_ids_for_atoms(&evidence_atom_ids, atoms);
        return Some(UnifiedAnswerDraft {
            text,
            draft_mode: "synthesized",
            rendered_claims: vec![AnswerClaimPlanReceipt {
                claim_id: "claim-1".to_string(),
                claim_type: "fact".to_string(),
                text: Some(claim_text),
                required_slots: plan.required_slots.clone(),
                evidence_ids,
                evidence_atom_ids,
                evidence_source_anchors,
                derivation,
                support_status: "planned".to_string(),
            }],
        });
    }
    None
}

fn build_participant_relationship_actor_conflict_partial(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
) -> Option<UnifiedAnswerDraft> {
    if !plan.participant_relationship_qualifier {
        return None;
    }
    let subject = plan.subject.as_deref()?;
    let answer_atom = atoms.iter().find(|atom| {
        atom.supports_slots
            .iter()
            .any(|slot| slot == "relationship_actor_conflict")
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact")
    })?;
    let actor = answer_atom.speaker.as_deref()?;
    let relation_context = atoms
        .iter()
        .filter(|candidate| {
            candidate.atom_id != answer_atom.atom_id
                && candidate.memory_id == answer_atom.memory_id
                && candidate.ordinal <= answer_atom.ordinal
                && answer_atom.ordinal.saturating_sub(candidate.ordinal) <= 4
                && candidate.kind.grounding_role() == GroundingRole::Original
                && candidate
                    .speaker
                    .as_deref()
                    .is_some_and(|candidate_speaker| {
                        normalize_person_key(candidate_speaker) == normalize_person_key(actor)
                    })
                && plan
                    .qualifier_terms
                    .iter()
                    .all(|term| text_matches_lexical_term(&candidate.text.to_lowercase(), term))
        })
        .max_by_key(|candidate| candidate.ordinal);
    let content = clean_answer_statement(&answer_atom.text);
    let citations = answer_atom_citations(plan, answer_atom, atoms);
    let relation = relation_context
        .map(|context| format!(" \"{}\"", clean_answer_statement(&context.text)))
        .unwrap_or_default();
    let claim_text = format!("The available evidence attributes this interaction to {actor}, not {subject}:{relation} \"{content}\"");
    let mut claim_atom_ids = vec![answer_atom.atom_id.clone()];
    if let Some(context) = relation_context {
        claim_atom_ids.push(context.atom_id.clone());
    }
    let (evidence_atom_ids, evidence_source_anchors, mut derivation) =
        claim_evidence_bindings_for_atom_ids(atoms, &claim_atom_ids);
    derivation.push("relationship_actor_conflict".to_string());
    derivation.sort();
    derivation.dedup();
    let evidence_ids = selected_memory_ids_for_atoms(&evidence_atom_ids, atoms);
    Some(UnifiedAnswerDraft {
        text: format!("Direct answer: {claim_text} {citations}"),
        draft_mode: "partial",
        rendered_claims: vec![AnswerClaimPlanReceipt {
            claim_id: "claim-1".to_string(),
            claim_type: "fact".to_string(),
            text: Some(claim_text),
            required_slots: plan.required_slots.clone(),
            evidence_ids,
            evidence_atom_ids,
            evidence_source_anchors,
            derivation,
            support_status: "planned".to_string(),
        }],
    })
}

fn build_collective_participant_attribution_partial(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
) -> Option<UnifiedAnswerDraft> {
    let subject = plan.subject.as_deref()?;
    let answer_atom = atoms.iter().find(|atom| {
        atom.supports_slots
            .iter()
            .any(|slot| slot == "collective_participant_context_bridge")
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact")
    })?;
    let actor = answer_atom.speaker.as_deref()?;
    if normalize_person_key(actor) == normalize_person_key(subject) {
        return None;
    }
    let content = clean_answer_statement(&answer_atom.text);
    let claim_text = format!(
        "Partial answer: The available evidence attributes this event to {actor}, not {subject}: \"{content}\""
    );
    let citations = answer_atom_citations(plan, answer_atom, atoms);
    let evidence_ids = vec![answer_atom.memory_id];
    let (evidence_atom_ids, evidence_source_anchors, derivation) =
        claim_evidence_bindings_from_atoms(atoms, &evidence_ids);
    Some(UnifiedAnswerDraft {
        text: format!("{claim_text} {citations}"),
        draft_mode: "partial",
        rendered_claims: vec![AnswerClaimPlanReceipt {
            claim_id: "claim-1".to_string(),
            claim_type: "fact".to_string(),
            text: Some(claim_text),
            required_slots: plan.required_slots.clone(),
            evidence_ids,
            evidence_atom_ids,
            evidence_source_anchors,
            derivation,
            support_status: "planned".to_string(),
        }],
    })
}

fn build_preference_recommendation_answer(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    chinese: bool,
) -> Option<UnifiedAnswerDraft> {
    if !plan.recommendation_requested {
        return None;
    }
    let capability = atoms.iter().find(|atom| {
        atom.supports_slots
            .iter()
            .any(|slot| slot == "recommendation_capability")
    })?;
    let topic = atoms.iter().find(|atom| {
        atom.supports_slots
            .iter()
            .any(|slot| slot == "recommendation_topic")
    })?;
    let capability_label = recommendation_capability_label(&capability.text)?;
    let topic_label = plan.recommendation_topic_terms.first()?;
    let recommendation = if capability_label == "writing" {
        format!("Writing a {topic_label} blog")
    } else {
        format!("A {capability_label} project focused on {topic_label}")
    };
    let evidence_ids = [capability.memory_id, topic.memory_id]
        .into_iter()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let citations = [capability, topic]
        .into_iter()
        .map(|atom| answer_atom_citations(plan, atom, atoms))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(" ");
    let text = if chinese {
        format!("推断建议：{recommendation}，因为它把已证实的 {capability_label} 能力与 {topic_label} 兴趣结合起来。{citations}")
    } else {
        format!("Likely recommendation: {recommendation} would fit because it combines the demonstrated {capability_label} practice with the {topic_label} interest. {citations}")
    };
    let (evidence_atom_ids, evidence_source_anchors, derivation) =
        claim_evidence_bindings_from_atoms(atoms, &evidence_ids);
    Some(UnifiedAnswerDraft {
        text: text.clone(),
        draft_mode: "inferred",
        rendered_claims: vec![AnswerClaimPlanReceipt {
            claim_id: "claim-1".to_string(),
            claim_type: "inference".to_string(),
            text: Some(text),
            required_slots: plan.required_slots.clone(),
            evidence_ids,
            evidence_atom_ids,
            evidence_source_anchors,
            derivation,
            support_status: "planned".to_string(),
        }],
    })
}

fn build_direct_aspect_set_answer(
    plan: &QueryPlan,
    selected: &[&EvidenceAtom],
    atoms: &[EvidenceAtom],
    chinese: bool,
) -> Option<String> {
    if chinese || plan.answer_shape != AnswerShape::AspectSet {
        return None;
    }
    let subject = plan.subject.as_deref()?;
    let mut facets = Vec::new();
    let mut seen = HashSet::new();
    for atom in selected {
        let facet = aspect_facet_from_statement(&atom.text)?;
        if seen.insert(normalize_for_dedup(&facet)) {
            facets.push(facet);
        }
    }
    if facets.is_empty() {
        return None;
    }
    let citations = selected
        .iter()
        .map(|atom| answer_atom_citations(plan, atom, atoms))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(" ");
    Some(format!(
        "Direct answer: {subject} highlighted these positive aspects: {}. {citations}",
        facets.join("; ")
    ))
}

fn aspect_facet_from_statement(value: &str) -> Option<String> {
    let statement = clean_answer_statement(value);
    if !statement.is_ascii() {
        return None;
    }
    let lower = statement.to_ascii_lowercase();
    for marker in [
        " like ",
        " likes ",
        " love ",
        " loves ",
        " admire ",
        " admires ",
        " enjoy ",
        " enjoys ",
        " appreciate ",
        " appreciates ",
        " value ",
        " values ",
    ] {
        if let Some(index) = lower.find(marker) {
            return normalize_aspect_facet(&statement[index + marker.len()..]);
        }
    }
    for marker in [
        "like ",
        "likes ",
        "love ",
        "loves ",
        "admire ",
        "admires ",
        "enjoy ",
        "enjoys ",
        "appreciate ",
        "appreciates ",
        "value ",
        "values ",
    ] {
        if let Some(rest) = lower.strip_prefix(marker) {
            let offset = lower.len() - rest.len();
            return normalize_aspect_facet(&statement[offset..]);
        }
    }
    let possessive_index = lower.find("'s ")?;
    let tail = &statement[possessive_index + 3..];
    let tail_lower = tail.to_ascii_lowercase();
    for marker in [
        " are incredible",
        " are amazing",
        " are great",
        " are impressive",
        " are remarkable",
        " are excellent",
        " are awesome",
        " are fantastic",
        " are wonderful",
        " is incredible",
        " is amazing",
        " is great",
        " is impressive",
        " is remarkable",
        " is excellent",
        " is awesome",
        " is fantastic",
        " is wonderful",
    ] {
        if let Some(index) = tail_lower.find(marker) {
            return normalize_aspect_facet(&tail[..index]);
        }
    }
    None
}

fn normalize_aspect_facet(value: &str) -> Option<String> {
    let mut facet = value
        .trim_matches(|character: char| {
            character.is_whitespace() || matches!(character, '.' | '!' | '?' | ';' | ':' | ',')
        })
        .to_string();
    loop {
        let lower = facet.to_ascii_lowercase();
        let Some(prefix) = [
            "his ",
            "her ",
            "their ",
            "its ",
            "the ",
            "moments of ",
            "qualities of ",
            "aspects of ",
        ]
        .iter()
        .find(|prefix| lower.starts_with(*prefix)) else {
            break;
        };
        facet = facet[prefix.len()..].trim().to_string();
    }
    (!facet.is_empty() && !informative_terms(&facet).is_empty()).then_some(facet)
}
fn grounded_inference_confidence(
    plan: &QueryPlan,
    selected: &[&EvidenceAtom],
    coverage: &[AnswerSlotCoverageReceipt],
) -> &'static str {
    let source_count = selected
        .iter()
        .map(|atom| atom.memory_id)
        .collect::<BTreeSet<_>>()
        .len();
    let slots_complete = coverage.iter().all(|slot| slot.status == "covered");

    match plan.inference_type {
        // An alternative history is never observed directly. More independent
        // evidence can strengthen its basis, but cannot turn it into a fact.
        InferenceType::Counterfactual if source_count >= 3 && slots_complete => "moderate",
        InferenceType::Counterfactual => "low",
        InferenceType::Likely if source_count >= 3 && slots_complete => "high",
        InferenceType::Likely if source_count >= 2 => "moderate",
        InferenceType::Likely => "low",
        InferenceType::Direct => "high",
    }
}

fn localized_inference_confidence(confidence: &str, chinese: bool) -> &'static str {
    if !chinese {
        return match confidence {
            "high" => "high",
            "moderate" => "moderate",
            _ => "low",
        };
    }

    match confidence {
        "high" => "高",
        "moderate" => "中等",
        _ => "低",
    }
}

fn build_grounded_inference_answer(
    plan: &QueryPlan,
    selected: &[&EvidenceAtom],
    atoms: &[EvidenceAtom],
    coverage: &[AnswerSlotCoverageReceipt],
    chinese: bool,
) -> Option<String> {
    if plan.inference_type == InferenceType::Direct {
        return None;
    }

    let clauses = selected
        .iter()
        .map(|atom| {
            format!(
                "{} {}",
                render_answer_statement(plan, atom),
                answer_atom_citations(plan, atom, atoms)
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    if clauses.is_empty() {
        return None;
    }

    let confidence = localized_inference_confidence(
        grounded_inference_confidence(plan, selected, coverage),
        chinese,
    );
    Some(match plan.inference_type {
        InferenceType::Likely if chinese => format!(
            "推断回答（{confidence}置信度）：基于已引用的原始证据，{clauses}"
        ),
        InferenceType::Likely => format!(
            "Likely answer ({confidence} confidence): Based on the cited original evidence, {clauses}"
        ),
        InferenceType::Counterfactual if chinese => format!(
            "反事实评估（{confidence}置信度）：已引用证据表明{clauses}。替代结果没有被直接记录，因此只能作为受限推断。"
        ),
        InferenceType::Counterfactual => format!(
            "Counterfactual assessment ({confidence} confidence): The cited evidence establishes {clauses}. The alternative outcome is not directly recorded, so this remains a bounded inference."
        ),
        InferenceType::Direct => return None,
    })
}

fn build_direct_location_answer(
    plan: &QueryPlan,
    selected: &[&EvidenceAtom],
    atoms: &[EvidenceAtom],
    chinese: bool,
) -> Option<String> {
    if plan.answer_target != AnswerTarget::Location
        || plan.cardinality != Cardinality::Single
        || plan.inference_type != InferenceType::Direct
        || !location_query_requires_subject_presence(plan)
        || selected.len() != 1
    {
        return None;
    }
    let atom = selected.first()?;
    if !atom.supports_slots.iter().any(|slot| slot == "location") {
        return None;
    }
    let subject = plan.subject.as_deref()?;
    let location = extract_location_answer_value(plan, &atom.text)?;
    let citation = answer_atom_citations(plan, atom, atoms);
    if chinese {
        Some(format!("{subject} 当时在 {location}。 {citation}"))
    } else {
        Some(format!("{subject} was in {location}. {citation}"))
    }
}

fn build_multi_entity_context_partial(
    query: &str,
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    answer_nodes: Option<usize>,
) -> Option<UnifiedAnswerDraft> {
    if !is_multi_entity_inference(plan) || atoms.iter().any(atom_is_answer_bearing) {
        return None;
    }
    let subject = plan.subject.as_deref()?;
    let mut selected = Vec::new();
    let limit = answer_nodes
        .unwrap_or(plan.named_entities.len())
        .clamp(1, 64);
    // Pick one source-backed premise per entity, respecting the user's budget.
    for entity in &plan.named_entities {
        let slot = named_entity_slot_name(entity);
        if let Some(atom) = atoms.iter().find(|atom| {
            atom_is_inference_entity_context(atom) && atom.supports_slots.contains(&slot)
        }) {
            if !selected
                .iter()
                .any(|selected: &&EvidenceAtom| selected.atom_id == atom.atom_id)
            {
                selected.push(atom);
            }
        }
        if selected.len() >= limit {
            break;
        }
    }
    if selected.is_empty() {
        return None;
    }
    let chinese = query
        .chars()
        .any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
    let entities = plan
        .named_entities
        .join(if chinese { "、" } else { " and " });
    let target = if plan.object_terms.is_empty() {
        plan.answer_target.as_str().to_string()
    } else {
        plan.object_terms.join(", ")
    };
    let missing_entities = plan
        .named_entities
        .iter()
        .filter(|entity| {
            let slot = named_entity_slot_name(entity);
            !atoms.iter().any(|atom| {
                atom_is_inference_entity_context(atom) && atom.supports_slots.contains(&slot)
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    let mut text = if chinese {
        format!("部分回答：现有证据不足以确定同时符合 {entities} 的 {target}。已知 {subject} 的相关情况：")
    } else {
        format!("Partial answer: The available evidence does not establish {target} connecting {entities} for {subject}. Known context:")
    };
    let mut rendered_claims = Vec::new();
    for atom in selected {
        let statement = render_answer_statement(plan, atom);
        text.push_str(&format!(
            "\n- {statement} {}",
            answer_atom_citations(plan, atom, atoms)
        ));
        let evidence_ids = vec![atom.memory_id];
        let (evidence_atom_ids, evidence_source_anchors, derivation) =
            claim_evidence_bindings_from_atoms(atoms, &evidence_ids);
        rendered_claims.push(AnswerClaimPlanReceipt {
            claim_id: format!("claim-{}", rendered_claims.len() + 1),
            claim_type: "fact".to_string(),
            text: Some(statement),
            required_slots: atom
                .supports_slots
                .iter()
                .filter(|slot| slot.starts_with("named_entity:") || slot.as_str() == "subject")
                .cloned()
                .collect(),
            evidence_ids,
            evidence_atom_ids,
            evidence_source_anchors,
            derivation,
            support_status: "planned".to_string(),
        });
    }
    if !missing_entities.is_empty() {
        text.push_str(&if chinese {
            format!("\n缺少相关证据：{}。", missing_entities.join("、"))
        } else {
            format!("\nMissing context: {}.", missing_entities.join(", "))
        });
    }
    Some(UnifiedAnswerDraft {
        text,
        draft_mode: "partial",
        rendered_claims,
    })
}

struct DurationProof {
    first_label: String,
    second_label: String,
    first_date: chrono::NaiveDate,
    second_date: chrono::NaiveDate,
    first_atoms: Vec<String>,
    second_atoms: Vec<String>,
}

impl DurationProof {
    fn all_atom_ids(&self) -> Vec<String> {
        self.first_atoms
            .iter()
            .chain(&self.second_atoms)
            .cloned()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn claims(&self, chinese: bool) -> Vec<(String, Vec<String>)> {
        let days = (self.second_date - self.first_date).num_days().abs();
        let texts = if chinese {
            vec![
                format!("事件一：{}，日期 {}。", self.first_label, self.first_date),
                format!("事件二：{}，日期 {}。", self.second_label, self.second_date),
                format!("相隔：{days} 天。"),
                format!("若首尾两天都计入：{} 天。", days + 1),
            ]
        } else {
            vec![
                format!("First event: {} on {}.", self.first_label, self.first_date),
                format!(
                    "Second event: {} on {}.",
                    self.second_label, self.second_date
                ),
                format!("Elapsed time: {days} days."),
                format!(
                    "Inclusive count: {} days when both boundary dates are counted.",
                    days + 1
                ),
            ]
        };
        texts
            .into_iter()
            .enumerate()
            .map(|(index, text)| {
                (
                    text,
                    match index {
                        0 => self.first_atoms.clone(),
                        1 => self.second_atoms.clone(),
                        _ => self.all_atom_ids(),
                    },
                )
            })
            .collect()
    }
}

fn duration_operand_matches(atom: &EvidenceAtom, clause: &str) -> bool {
    let terms = claim_content_terms(&informative_terms(clause), &[]);
    atom.kind.grounding_role() == GroundingRole::Original
        && !terms.is_empty()
        && terms
            .iter()
            .filter(|term| term.as_str() != "at")
            .all(|term| text_matches_term(&atom.text.to_lowercase(), term))
}

fn duration_proof(plan: &QueryPlan, atoms: &[EvidenceAtom]) -> Option<DurationProof> {
    let (first, second) = plan.duration_clauses.as_ref()?;
    let resolve = |clause: &str| -> Option<(chrono::NaiveDate, Vec<String>)> {
        let mut candidates = Vec::new();
        for atom in atoms
            .iter()
            .filter(|atom| atom.kind.grounding_role() == GroundingRole::Original)
        {
            let lower = atom.text.to_lowercase();
            if !duration_operand_matches(atom, clause)
                || plan
                    .subject
                    .as_deref()
                    .is_some_and(|subject| !atom_has_verified_fact_subject(atom, subject))
                || !extract_negation_terms(&lower).is_empty()
                || source_condition_clause(&lower).is_some()
                || !statement_records_observed_action(&atom.text)
            {
                continue;
            }
            let session_dates = atoms
                .iter()
                .filter(|context| {
                    context.memory_id == atom.memory_id && context.kind == EvidenceKind::SessionDate
                })
                .filter_map(|context| {
                    crate::memory_engine_summary::source_session_date(&context.text)
                        .map(|date| (date, context))
                })
                .collect::<Vec<_>>();
            let years = session_dates
                .iter()
                .map(|(date, _)| date.year())
                .collect::<BTreeSet<_>>();
            let year = if years.len() == 1 { *years.first()? } else { 0 };
            let dates = crate::memory_engine_summary::temporal_date_anchors(&atom.text, year)
                .into_iter()
                .filter(|(_, _, _, metadata)| !metadata)
                .map(|(date, _, _, _)| date)
                .collect::<BTreeSet<_>>();
            if dates.len() > 1 {
                return None;
            }
            if dates.is_empty() {
                continue;
            }
            if dates.first()?.year() == 0 {
                return None;
            }
            let inferred_year = crate::memory_engine_summary::temporal_date_anchors(&atom.text, 0)
                .iter()
                .any(|(date, _, _, _)| date.year() == 0);
            if inferred_year
                && [
                    "last year",
                    "previous year",
                    "next year",
                    "last season",
                    "previous season",
                    "years ago",
                    "year ago",
                    "前年",
                    "去年",
                    "明年",
                ]
                .iter()
                .any(|phrase| lower.contains(phrase))
            {
                return None;
            }
            let mut ids = vec![atom.atom_id.clone()];
            ids.extend(
                session_dates
                    .iter()
                    .map(|(_, context)| context.atom_id.clone()),
            );
            candidates.push((*dates.first()?, ids));
        }
        let dates = candidates
            .iter()
            .map(|(date, _)| *date)
            .collect::<BTreeSet<_>>();
        if dates.len() != 1 {
            return None;
        }
        candidates.into_iter().next()
    };
    let (first_date, first_atoms) = resolve(first)?;
    let (second_date, second_atoms) = resolve(second)?;
    if first_atoms.first() == second_atoms.first() {
        return None;
    }
    Some(DurationProof {
        first_label: first.clone(),
        second_label: second.clone(),
        first_date,
        second_date,
        first_atoms,
        second_atoms,
    })
}

fn build_unified_answer(
    query: &str,
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    coverage: &[AnswerSlotCoverageReceipt],
    answer_nodes: Option<usize>,
    detail_level: &str,
) -> Option<UnifiedAnswerDraft> {
    if plan.duration_clauses.is_some()
        && coverage
            .iter()
            .any(|slot| slot.slot == "proof_scope" && slot.status != "covered")
    {
        return None;
    }
    if let Some(proof) = duration_proof(plan, atoms) {
        let chinese = query.chars().any(is_cjk);
        let rendered_claims = proof
            .claims(chinese)
            .into_iter()
            .enumerate()
            .map(|(index, (text, ids))| {
                let (atom_ids, anchors, mut derivation) =
                    claim_evidence_bindings_for_atom_ids(atoms, &ids);
                derivation.push("date_difference_v1".into());
                AnswerClaimPlanReceipt {
                    claim_id: format!("claim-{}", index + 1),
                    claim_type: "calculation".into(),
                    text: Some(text),
                    required_slots: plan.required_slots.clone(),
                    evidence_ids: selected_memory_ids_for_atoms(&atom_ids, atoms),
                    evidence_atom_ids: atom_ids,
                    evidence_source_anchors: anchors,
                    derivation,
                    support_status: "planned".into(),
                }
            })
            .collect::<Vec<_>>();
        let lines = rendered_claims
            .iter()
            .map(|claim| {
                format!(
                    "- {} {}",
                    claim.text.as_deref().unwrap_or_default(),
                    claim
                        .evidence_ids
                        .iter()
                        .map(|id| format!("[memory#{id}]"))
                        .collect::<Vec<_>>()
                        .join(" ")
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        return Some(UnifiedAnswerDraft {
            text: format!(
                "{}\n\n{lines}",
                if chinese {
                    "时间推理："
                } else {
                    "Temporal reasoning:"
                }
            ),
            draft_mode: if coverage.iter().any(|slot| slot.status != "covered") {
                "partial"
            } else {
                "structured"
            },
            rendered_claims,
        });
    }
    if let Some(partial) = build_multi_entity_context_partial(query, plan, atoms, answer_nodes) {
        return Some(partial);
    }
    let missing_critical = coverage.iter().any(|slot| {
        slot.status == "missing"
            && (slot.slot.starts_with("person:")
                || matches!(
                    slot.slot.as_str(),
                    "answer_bearing_fact"
                        | "subject"
                        | "action"
                        | "action_object_complement"
                        | "observed_action_fact"
                        | "categorical_attribute_value"
                        | "duration_value"
                        | "list_value"
                        | "list_participant_scope"
                        | "achievement_extremum"
                        | "object"
                        | "answer_class"
                        | "qualifier_context"
                        | "location"
                        | "event_time"
                        | "negation"
                        | "condition"
                        | "requested_modality"
                ))
    });
    let mut partial = coverage.iter().any(|slot| slot.status != "covered")
        || plan.inference_type == InferenceType::Counterfactual;
    if missing_critical {
        if let Some(partial) = build_participant_relationship_actor_conflict_partial(plan, atoms) {
            return Some(partial);
        }
        if let Some(partial) = build_collective_participant_attribution_partial(plan, atoms) {
            return Some(partial);
        }
    }
    if !missing_critical {
        if let Some(temporal) = build_session_anchored_temporal_answer(query, plan, atoms) {
            return Some(temporal);
        }
    }
    if !missing_critical && plan.cardinality == Cardinality::Count {
        let chinese = query
            .chars()
            .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
        return build_count_unified_answer(plan, atoms, partial, chinese);
    }
    let mut selected = Vec::<&EvidenceAtom>::new();
    if !missing_critical {
        let query_is_chinese = query
            .chars()
            .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
        if let Some(recommendation) =
            build_preference_recommendation_answer(plan, atoms, query_is_chinese)
        {
            return Some(recommendation);
        }
    }
    let mut seen = HashSet::new();
    let mut selected_context_groups = HashSet::new();
    let available_answer_nodes = atoms
        .iter()
        .filter(|atom| {
            atom.kind.grounding_role() == GroundingRole::Original
                && atom
                    .supports_slots
                    .iter()
                    .any(|slot| slot == "answer_bearing_fact")
        })
        .count()
        .max(1);
    let primary_score = atoms
        .iter()
        .find(|atom| {
            matches!(
                atom.kind,
                EvidenceKind::Statement | EvidenceKind::Event | EvidenceKind::NativeMedia
            ) && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact")
        })
        .map(|atom| atom.score)?;
    let list_score_delta = match plan.answer_shape {
        // A question about qualities, preferences, or other aspects is not
        // answered by a single highest-scoring sentence. Preserve relevant,
        // complementary evidence even when the wording differs substantially.
        AnswerShape::AspectSet | AnswerShape::ActionSet => 3.25,
        AnswerShape::List | AnswerShape::BoundedList if plan.grounded_list_kind.is_some() => 3.25,
        _ if plan.list_scope == "exhaustive" => 1.25,
        AnswerShape::List | AnswerShape::BoundedList if plan.achievement_extremum => 3.25,
        _ => 0.75,
    };
    let high_confidence_answer_nodes = atoms
        .iter()
        .filter(|atom| {
            matches!(
                atom.kind,
                EvidenceKind::Statement | EvidenceKind::Event | EvidenceKind::NativeMedia
            ) && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact")
                && (atom.score >= primary_score - list_score_delta
                    // A record/personal-best list can contain complementary facts with very
                    // different lexical scores. Once the evidence plan explicitly admits one,
                    // its answer node must survive the renderer's automatic node budget too.
                    || (plan.achievement_extremum
                        && atom
                            .supports_slots
                            .iter()
                            .any(|slot| slot == "achievement_extremum")))
        })
        .count()
        .max(1);
    let automatic_nodes = match plan.cardinality {
        Cardinality::Count => available_answer_nodes,
        Cardinality::List => high_confidence_answer_nodes,
        Cardinality::Single
            if plan.answer_shape == AnswerShape::ActionSet
                || plan.inference_type != InferenceType::Direct =>
        {
            available_answer_nodes.min(4)
        }
        Cardinality::Single
            if plan.answer_target == AnswerTarget::Manner && !plan.qualifier_terms.is_empty() =>
        {
            2
        }
        Cardinality::Single
            if matches!(
                plan.answer_target,
                AnswerTarget::Description
                    | AnswerTarget::Object
                    | AnswerTarget::Entity
                    | AnswerTarget::Location
            ) =>
        {
            1
        }
        Cardinality::Single => match detail_level {
            "concise" => 2,
            "balanced" => 3,
            "detailed" => 4,
            "custom" => 3,
            _ => 3,
        },
    };
    let node_limit = answer_nodes.unwrap_or(automatic_nodes).clamp(1, 64);
    let score_floor = primary_score
        - match plan.cardinality {
            Cardinality::Single
                if plan.answer_target == AnswerTarget::Manner
                    && !plan.qualifier_terms.is_empty() =>
            {
                3.25
            }
            Cardinality::Single
                if plan.answer_shape == AnswerShape::ActionSet
                    || plan.inference_type != InferenceType::Direct =>
            {
                3.25
            }
            Cardinality::Single => 1.0,
            Cardinality::List => list_score_delta,
            Cardinality::Count => 1.75,
        };
    for atom in atoms.iter().filter(|atom| {
        matches!(
            atom.kind,
            EvidenceKind::Statement | EvidenceKind::Event | EvidenceKind::NativeMedia
        ) && atom
            .supports_slots
            .iter()
            .any(|slot| slot == "answer_bearing_fact")
    }) {
        let explicit_complementary_extremum = plan.achievement_extremum
            && atom
                .supports_slots
                .iter()
                .any(|slot| slot == "achievement_extremum");
        if answer_nodes.is_none()
            && !selected.is_empty()
            && atom.score < score_floor
            && !explicit_complementary_extremum
        {
            break;
        }
        let context_key = (atom.memory_id, atom.context_group);
        if answer_nodes.is_none()
            && plan.cardinality == Cardinality::Single
            && atom.context_group != usize::MAX
            && selected_context_groups.contains(&context_key)
        {
            continue;
        }
        if plan.cardinality == Cardinality::List
            && atom.context_group != usize::MAX
            && selected.iter().any(|selected_atom| {
                selected_atom.memory_id == atom.memory_id
                    && selected_atom.context_group == atom.context_group
            })
            && is_referential_list_continuation(&atom.text)
        {
            continue;
        }
        let key = normalize_for_dedup(&atom.text);
        if key.chars().count() >= 4 && seen.insert(key) {
            selected.push(atom);
            if atom.context_group != usize::MAX {
                selected_context_groups.insert(context_key);
            }
        }
        if selected.len() >= node_limit {
            break;
        }
    }
    if selected.is_empty() || missing_critical {
        return None;
    }
    if selected.len() < available_answer_nodes && matches!(plan.cardinality, Cardinality::List) {
        partial = true;
    }

    // Every generated branch must nominate structured Claims. In particular,
    // a scalar direct answer cannot rely on its rendered prose being parsed
    // back into a Claim: delivery deliberately emits only independently
    // rebound, accepted Claims.
    let rendered_claims = selected
        .iter()
        .enumerate()
        .map(|(index, atom)| {
            let mut evidence_ids = evidence_ids_for_claim_atom(atoms, atom);
            if !plan.qualifier_terms.is_empty() {
                evidence_ids.extend(evidence_ids_for_slot(atoms, "qualifier_context"));
            }
            evidence_ids.sort_unstable();
            evidence_ids.dedup();
            let (evidence_atom_ids, evidence_source_anchors, derivation) =
                claim_evidence_bindings_from_atoms(atoms, &evidence_ids);
            AnswerClaimPlanReceipt {
                claim_id: format!("claim-{}", index + 1),
                claim_type: if plan.inference_type == InferenceType::Direct {
                    "fact".to_string()
                } else {
                    "inference".to_string()
                },
                text: Some(render_answer_statement(plan, atom)),
                required_slots: plan.required_slots.clone(),
                evidence_ids,
                evidence_atom_ids,
                evidence_source_anchors,
                derivation,
                support_status: "planned".to_string(),
            }
        })
        .collect();

    let chinese = query
        .chars()
        .any(|character| ('\u{4e00}'..='\u{9fff}').contains(&character));
    let text = if let Some(inferred_answer) =
        build_grounded_inference_answer(plan, &selected, atoms, coverage, chinese)
    {
        inferred_answer
    } else if let Some(aspect_answer) =
        build_direct_aspect_set_answer(plan, &selected, atoms, chinese)
    {
        aspect_answer
    } else if let Some(location_answer) =
        build_direct_location_answer(plan, &selected, atoms, chinese)
    {
        location_answer
    } else if plan.cardinality == Cardinality::List {
        let lines = selected
            .iter()
            .map(|atom| {
                format!(
                    "- {} {}",
                    render_answer_statement(plan, atom),
                    answer_atom_citations(plan, atom, atoms)
                )
            })
            .collect::<Vec<_>>()
            .join("\n");
        let completeness_notice = if partial && plan.list_scope == "open" {
            if chinese {
                "\n\n现有证据确认以上条目，但不足以证明列表完整。"
            } else {
                "\n\nThe available evidence confirms these entries but does not establish a complete list."
            }
        } else {
            ""
        };
        if chinese {
            format!("直接回答：\n{lines}{completeness_notice}")
        } else {
            format!("Direct answer:\n{lines}{completeness_notice}")
        }
    } else {
        let clauses = selected
            .iter()
            .map(|atom| {
                format!(
                    "{} {}",
                    render_answer_statement(plan, atom),
                    answer_atom_citations(plan, atom, atoms)
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        if chinese {
            format!("直接回答：{clauses}")
        } else {
            format!("Direct answer: {clauses}")
        }
    };
    Some(UnifiedAnswerDraft {
        text,
        draft_mode: match plan.inference_type {
            InferenceType::Direct if partial => "partial",
            InferenceType::Direct => "synthesized",
            InferenceType::Likely if partial => "partial_inferred",
            InferenceType::Likely => "inferred",
            InferenceType::Counterfactual => "partial_inferred",
        },
        rendered_claims,
    })
}

fn build_count_unified_answer(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    partial: bool,
    chinese: bool,
) -> Option<UnifiedAnswerDraft> {
    let count_events = count_event_set(plan, atoms, None);
    if !count_events.is_proven() {
        return None;
    }
    let evidence_atoms = count_events
        .atom_ids
        .iter()
        .filter_map(|atom_id| atoms.iter().find(|atom| atom.atom_id == *atom_id))
        .collect::<Vec<_>>();
    let citations = evidence_atoms
        .iter()
        .map(|atom| answer_atom_citations(plan, atom, atoms))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join(" ");
    let mut shared_condition = None::<String>;
    let mut conditions = Vec::new();
    for atom in &evidence_atoms {
        let lower = atom.text.to_lowercase();
        let condition = claim_event_condition_clause(plan, &lower)
            .and_then(|_| source_condition_parts(&lower))
            .map(|(_, marker, _, tail)| {
                format!("{marker} {}", tail.trim().trim_end_matches(['.', '。']))
            });
        conditions.push(condition);
    }
    if let Some(first) = conditions.first() {
        if !conditions.iter().all(|condition| condition == first) {
            return None;
        }
        shared_condition = first.clone();
    }
    let suffix = shared_condition
        .map(|value| format!(" {value}"))
        .unwrap_or_default();
    let claim_text = if chinese {
        format!("直接回答：{}{suffix}。", count_events.display_value)
    } else {
        format!("Direct answer: {}{suffix}.", count_events.display_value)
    };
    let text = if citations.is_empty() {
        claim_text.clone()
    } else {
        format!("{claim_text} {citations}")
    };
    let (evidence_atom_ids, evidence_source_anchors, mut derivation) =
        claim_evidence_bindings_for_atom_ids(atoms, &count_events.atom_ids);
    derivation.extend(count_events.derivation.clone());
    derivation.sort();
    derivation.dedup();
    Some(UnifiedAnswerDraft {
        text,
        draft_mode: if partial { "partial" } else { "synthesized" },
        rendered_claims: vec![AnswerClaimPlanReceipt {
            claim_id: "claim-1".to_string(),
            claim_type: "fact".to_string(),
            text: Some(claim_text),
            required_slots: plan.required_slots.clone(),
            evidence_ids: count_events.memory_ids,
            evidence_atom_ids,
            evidence_source_anchors,
            derivation,
            support_status: "planned".to_string(),
        }],
    })
}

fn count_event_set(
    plan: &QueryPlan,
    atoms: &[EvidenceAtom],
    candidate_ids: Option<&HashSet<u64>>,
) -> CountEventSet {
    let eligible = atoms
        .iter()
        .filter(|atom| {
            candidate_ids.is_none_or(|ids| ids.contains(&atom.memory_id))
                && count_atom_matches_plan(plan, atom)
        })
        .collect::<Vec<_>>();
    if eligible.is_empty() {
        return CountEventSet::default();
    }
    let session_dates = count_event_session_dates(atoms);
    if eligible
        .iter()
        .any(|atom| count_quantity_value(&atom.text).is_err())
    {
        return CountEventSet {
            derivation: vec!["count_quantity_scope_unresolved".into()],
            ..CountEventSet::default()
        };
    }
    if eligible.iter().any(|atom| {
        !atom_is_count_event_bridge(atom)
            && count_event_date_resolution(plan, atom, &session_dates) == CountEventDate::Ambiguous
    }) {
        // Do not drop this atom and count the remainder, or turn a multi-event
        // statement into the default single occurrence via an undated key.
        return CountEventSet {
            derivation: vec!["count_event_cardinality_unresolved".into()],
            ..CountEventSet::default()
        };
    }

    // Differing descriptions are not independent occurrences merely because
    // their keys differ. Without disjoint dates or an explicit before/after
    // distinction, leave the aggregate unproven.
    let mut comparisons = 0_usize;
    for (index, left) in eligible.iter().enumerate() {
        for right in eligible.iter().skip(index + 1) {
            comparisons += 1;
            if comparisons > 65_536 {
                return CountEventSet::default();
            }
            if atom_is_count_event_bridge(left)
                || atom_is_count_event_bridge(right)
                || count_event_signature(plan, left) != count_event_signature(plan, right)
            {
                continue;
            }
            let left_date = count_event_date(plan, left, &session_dates);
            let right_date = count_event_date(plan, right, &session_dates);
            if normalize_for_dedup(&left.text) == normalize_for_dedup(&right.text)
                && (left_date.is_some() && left_date == right_date
                    || left.memory_id == right.memory_id
                        && left
                            .source_anchor
                            .as_ref()
                            .zip(right.source_anchor.as_ref())
                            .is_some_and(|(a, b)| a.kind == b.kind && a.value == b.value))
            {
                continue;
            }
            if left_date
                .zip(right_date)
                .is_some_and(|(left, right)| left != right)
            {
                continue;
            }
            let left_context = count_event_context_discriminator(plan, left);
            let right_context = count_event_context_discriminator(plan, right);
            if left_context
                .as_deref()
                .zip(right_context.as_deref())
                .is_some_and(|(left, right)| {
                    let opposite = |a: &str, b: &str| {
                        a.strip_prefix("before:")
                            .zip(b.strip_prefix("after:"))
                            .is_some_and(|(a, b)| a == b)
                    };
                    opposite(left, right) || opposite(right, left)
                })
            {
                continue;
            }
            return CountEventSet::default();
        }
    }

    // An anaphoric bridge can extend the provenance of an event, but it is
    // never a new event by itself. Its identity is the exact anchor atom that
    // justified the bridge during evidence planning.
    let anchor_identities = eligible
        .iter()
        .filter(|atom| !atom_is_count_event_bridge(atom))
        .filter_map(|atom| {
            let identity = count_event_identity(plan, atom, &session_dates);
            (!identity.is_empty()).then(|| (atom.atom_id.clone(), identity))
        })
        .collect::<HashMap<_, _>>();
    let mut groups = BTreeMap::<String, Vec<&EvidenceAtom>>::new();
    for atom in eligible {
        let identity = if let Some(anchor_id) = count_event_bridge_anchor(atom) {
            let Some(anchor_identity) = anchor_identities.get(anchor_id) else {
                // A bridge without its exact anchor is only a retrieval hint;
                // it cannot enter a numerical proof.
                continue;
            };
            // Use the anchor's event identity exactly, so a referential
            // phrase contributes provenance without becoming a second event.
            format!("event:{anchor_identity}")
        } else {
            let identity = count_event_identity(plan, atom, &session_dates);
            if identity.is_empty() {
                continue;
            }
            format!("event:{identity}")
        };
        groups.entry(identity).or_default().push(atom);
    }
    if groups.is_empty() {
        return CountEventSet::default();
    }

    // Without a date, differently worded records for the same subject/action/
    // object could be duplicate descriptions of one event. Do not promote
    // them into separate numerical events without an explicit discriminator.
    let mut undated_variants = HashMap::<String, HashSet<String>>::new();
    for identity in groups.keys() {
        let Some(identity) = identity.strip_prefix("event:undated::") else {
            continue;
        };
        let Some((signature, text)) = identity.rsplit_once("::") else {
            return CountEventSet::default();
        };
        undated_variants
            .entry(signature.to_string())
            .or_default()
            .insert(text.to_string());
    }
    if undated_variants.values().any(|variants| variants.len() > 1) {
        return CountEventSet::default();
    }

    let mut atom_ids = BTreeSet::new();
    let mut memory_ids = BTreeSet::new();
    let mut total = 0_usize;
    let mut only_display = None::<String>;
    let mut derivation = vec!["count_event_set_v1".to_string()];
    let group_count = groups.len();
    for group in groups.values() {
        let explicit = group
            .iter()
            .filter_map(|atom| explicit_count_value_with_number(&atom.text))
            .collect::<Vec<_>>();
        let values = explicit
            .iter()
            .map(|(number, _)| *number)
            .collect::<BTreeSet<_>>();
        if values.len() > 1 {
            // Conflicting explicit quantities are not safely resolvable by a
            // record count. Leave the answer unproven instead of guessing.
            return CountEventSet::default();
        }
        let value = values.iter().next().copied().unwrap_or(1);
        let Some(sum) = total.checked_add(value) else {
            return CountEventSet::default();
        };
        total = sum;
        if group_count == 1 {
            only_display = explicit.first().map(|(_, display)| display.clone());
        }
        for atom in group {
            atom_ids.insert(atom.atom_id.clone());
            memory_ids.insert(atom.memory_id);
            if atom_is_count_event_bridge(atom) {
                derivation.push("count_event_bridge_is_not_a_new_event".to_string());
            }
        }
    }
    derivation.sort();
    derivation.dedup();
    CountEventSet {
        atom_ids: atom_ids.into_iter().collect(),
        memory_ids: memory_ids.into_iter().collect(),
        numeric_value: total,
        display_value: only_display.unwrap_or_else(|| total.to_string()),
        derivation,
    }
}

fn count_atom_matches_plan(plan: &QueryPlan, atom: &EvidenceAtom) -> bool {
    if atom.kind.grounding_role() != GroundingRole::Original
        || !atom_is_answer_bearing(atom)
        || !atom.supports_slots.iter().any(|slot| slot == "action")
        || !atom.supports_slots.iter().any(|slot| slot == "object")
    {
        return false;
    }
    if plan
        .subject
        .as_deref()
        .is_some_and(|subject| !atom_has_verified_fact_subject(atom, subject))
    {
        return false;
    }
    // Slot labels are generated during planning and can be inherited by a
    // neighbouring sentence. A count event additionally has to express the
    // requested predicate and object itself. The sole exception is an
    // explicit anaphoric bridge, which is retained only as provenance for its
    // exact anchor and never creates a new event.
    if !atom_is_count_event_bridge(atom) {
        let lower = atom.text.to_lowercase();
        if !plan.action_terms.is_empty()
            && !plan
                .action_terms
                .iter()
                .any(|term| text_matches_term(&lower, term))
        {
            return false;
        }
        if !plan.object_terms.is_empty()
            && !plan
                .object_terms
                .iter()
                .any(|term| text_matches_term(&lower, term))
        {
            return false;
        }
    }
    true
}

fn atom_is_count_event_bridge(atom: &EvidenceAtom) -> bool {
    atom.supports_slots
        .iter()
        .any(|slot| slot == "count_event_bridge")
}

fn count_event_bridge_anchor(atom: &EvidenceAtom) -> Option<&str> {
    atom.supports_slots
        .iter()
        .find_map(|slot| slot.strip_prefix("count_event_bridge_anchor:"))
}

fn count_event_session_dates(atoms: &[EvidenceAtom]) -> HashMap<u64, chrono::NaiveDate> {
    atoms
        .iter()
        .filter(|atom| atom.kind == EvidenceKind::SessionDate)
        .filter_map(|atom| {
            crate::memory_engine_summary::source_session_date(&atom.text)
                .map(|date| (atom.memory_id, date))
        })
        .collect()
}

fn count_event_signature(plan: &QueryPlan, atom: &EvidenceAtom) -> String {
    let lower = atom.text.to_lowercase();
    let action_terms = plan
        .action_terms
        .iter()
        .filter(|term| text_matches_term(&lower, term))
        .map(|term| normalize_for_dedup(term))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("+");
    let object_terms = plan
        .object_terms
        .iter()
        .filter(|term| text_matches_term(&lower, term))
        .map(|term| normalize_for_dedup(term))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("+");
    format!(
        "{}::{action_terms}::{object_terms}",
        plan.subject
            .as_deref()
            .map(normalize_person_key)
            .unwrap_or_default()
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CountEventDate {
    Missing,
    Exact(chrono::NaiveDate),
    Ambiguous,
}

fn count_event_date(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    session_dates: &HashMap<u64, chrono::NaiveDate>,
) -> Option<chrono::NaiveDate> {
    match count_event_date_resolution(plan, atom, session_dates) {
        CountEventDate::Exact(date) => Some(date),
        CountEventDate::Missing | CountEventDate::Ambiguous => None,
    }
}

fn count_event_date_resolution(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    session_dates: &HashMap<u64, chrono::NaiveDate>,
) -> CountEventDate {
    if plan.action_terms.is_empty() {
        return CountEventDate::Missing;
    }
    let scopes = count_event_time_clauses(&atom.text);
    let matching = scopes
        .into_iter()
        .filter(|scope| {
            let lower = scope.to_lowercase();
            plan.action_terms
                .iter()
                .any(|term| text_matches_term(&lower, term))
                && (plan.object_terms.is_empty()
                    || plan
                        .object_terms
                        .iter()
                        .any(|term| text_matches_term(&lower, term)))
        })
        .collect::<Vec<_>>();
    if matching.len() != 1 {
        return CountEventDate::Ambiguous;
    }
    let scope = matching[0];
    let lower = scope.to_lowercase();
    let action_mentions = word_regex()
        .find_iter(&lower)
        .filter(|word| {
            plan.action_terms
                .iter()
                .any(|action| text_matches_term(word.as_str(), action))
        })
        .count();
    if action_mentions > 1 {
        return CountEventDate::Ambiguous;
    }
    if [
        "last week",
        "next week",
        "this week",
        "last month",
        "this month",
        "next month",
    ]
    .iter()
    .any(|term| contains_bounded_ascii_phrase(&lower, term))
    {
        return if count_quantity_value(scope).is_ok_and(|value| value.is_some()) {
            CountEventDate::Missing
        } else {
            CountEventDate::Ambiguous
        };
    }
    let explicit = parse_temporal_dates(scope)
        .into_iter()
        .collect::<BTreeSet<_>>();
    if !explicit.is_empty() {
        return if explicit.len() == 1 {
            CountEventDate::Exact(*explicit.first().unwrap())
        } else {
            CountEventDate::Ambiguous
        };
    }
    let Some(session_date) = session_dates.get(&atom.memory_id) else {
        return CountEventDate::Missing;
    };
    let relative = exact_relative_event_dates(scope, *session_date)
        .into_iter()
        .collect::<BTreeSet<_>>();
    match relative.len() {
        0 => CountEventDate::Missing,
        1 => CountEventDate::Exact(*relative.first().unwrap()),
        _ => CountEventDate::Ambiguous,
    }
}

fn count_event_time_clauses(text: &str) -> Vec<&str> {
    // Delimiters inside native dates (e.g. May 12, 2023) are not clause breaks.
    // Keep simultaneous while/during adjuncts with the action. Competing
    // dates in that scope are rejected by count_event_date.
    let dates = date_regex()
        .find_iter(text)
        .map(|span| span.range())
        .collect::<Vec<_>>();
    static BOUNDARY: OnceLock<Regex> = OnceLock::new();
    let boundary = BOUNDARY.get_or_init(|| {
        Regex::new(r"(?i)[,;.!?\n，；。！？]|\b(?:after|before|because|following|if|unless|but)\b")
            .expect("event time clause boundary regex")
    });
    let mut clauses = Vec::new();
    let mut start = 0;
    for split in boundary.find_iter(text) {
        if dates.iter().any(|date| date.contains(&split.start())) {
            continue;
        }
        if start < split.start() {
            clauses.push(text[start..split.start()].trim());
        }
        start = split.end();
    }
    if start < text.len() {
        clauses.push(text[start..].trim());
    }
    clauses
}

fn count_event_context_discriminator(plan: &QueryPlan, atom: &EvidenceAtom) -> Option<String> {
    let lower = atom.text.to_lowercase();
    for marker in [
        " while ",
        " during ",
        " after ",
        " before ",
        " following ",
        " at ",
        "在",
        "期间",
        "之后",
        "之前",
    ] {
        for (start, _) in lower.match_indices(marker) {
            let context = &lower[start + marker.len()..];
            let context = context
                .split(|character: char| matches!(character, '.' | '!' | '?' | ';' | '\n'))
                .next()
                .unwrap_or_default();
            let terms = informative_terms(context)
                .into_iter()
                .filter(|term| {
                    !plan
                        .action_terms
                        .iter()
                        .any(|action| text_matches_term(term, action))
                        && !plan
                            .object_terms
                            .iter()
                            .any(|object| text_matches_term(term, object))
                        && plan.subject.as_deref().is_none_or(|subject| {
                            normalize_person_key(term) != normalize_person_key(subject)
                        })
                })
                .take(4)
                .collect::<Vec<_>>();
            if !terms.is_empty() {
                let relation = match marker.trim() {
                    "之前" => "before",
                    "之后" | "following" => "after",
                    other => other,
                };
                return Some(format!("{relation}:{}", terms.join("+")));
            }
        }
    }
    None
}

fn count_event_identity(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    session_dates: &HashMap<u64, chrono::NaiveDate>,
) -> String {
    let signature = count_event_signature(plan, atom);
    let text = normalize_for_dedup(&atom.text);
    if signature.is_empty() || text.is_empty() {
        return String::new();
    }
    match count_event_date_resolution(plan, atom, session_dates) {
        CountEventDate::Exact(date) => {
            let context = count_event_context_discriminator(plan, atom).unwrap_or_default();
            return format!("dated::{signature}::{date}::{context}");
        }
        CountEventDate::Ambiguous => return String::new(),
        CountEventDate::Missing => {}
    }
    if let Some(context) = count_event_context_discriminator(plan, atom) {
        return format!("context::{signature}::{context}");
    }
    format!("undated::{signature}::{text}")
}

fn explicit_count_value_with_number(value: &str) -> Option<(usize, String)> {
    count_quantity_value(value).ok().flatten()
}

fn chinese_count_integer(value: &str) -> Option<usize> {
    if let Ok(number) = value.parse::<usize>() {
        return Some(number);
    }
    let mut total = 0usize;
    let mut digit = 0usize;
    let mut last_unit = usize::MAX;
    for ch in value.chars() {
        match ch {
            '零' | '〇' => digit = 0,
            '一' => digit = 1,
            '二' | '两' => digit = 2,
            '三' => digit = 3,
            '四' => digit = 4,
            '五' => digit = 5,
            '六' => digit = 6,
            '七' => digit = 7,
            '八' => digit = 8,
            '九' => digit = 9,
            '十' | '百' | '千' | '万' => {
                let unit = match ch {
                    '十' => 10,
                    '百' => 100,
                    '千' => 1000,
                    _ => 10000,
                };
                if unit >= last_unit {
                    return None;
                }
                total = total.checked_add(digit.max(1).checked_mul(unit)?)?;
                last_unit = unit;
                digit = 0;
            }
            _ => return None,
        }
    }
    total.checked_add(digit)
}

fn count_quantity_value(value: &str) -> Result<Option<(usize, String)>, ()> {
    static COUNTS: OnceLock<Regex> = OnceLock::new();
    let lower = value.to_lowercase();
    let regex = COUNTS.get_or_init(|| Regex::new(
        r"\b(?:once|twice)\b|\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|[0-9]+)\s*(?:times?|occurrences?|instances?)\b|[零〇一二两三四五六七八九十百千万0-9]+\s*次"
    ).expect("complete count expression regex"));
    let mut mentions = Vec::new();
    for span in regex.find_iter(&lower) {
        let text = span.as_str();
        let number = if let Some(number) = text.strip_suffix('次') {
            chinese_count_integer(number.trim()).ok_or(())?
        } else {
            match text.split_whitespace().next().unwrap_or_default() {
                "once" | "one" => 1,
                "twice" | "two" => 2,
                "zero" => 0,
                "three" => 3,
                "four" => 4,
                "five" => 5,
                "six" => 6,
                "seven" => 7,
                "eight" => 8,
                "nine" => 9,
                "ten" => 10,
                _ => text
                    .chars()
                    .take_while(|ch| ch.is_ascii_digit())
                    .collect::<String>()
                    .parse::<usize>()
                    .map_err(|_| ())?,
            }
        };
        mentions.push((number, span));
    }
    if mentions.is_empty() {
        return Ok(None);
    }
    if mentions.len() == 1 {
        return Ok(Some((mentions[0].0, mentions[0].1.as_str().to_string())));
    }
    // Sum only explicitly disjoint local partitions, not a total plus its
    // components, alternatives, or repeated descriptions of the same event.
    let mut scopes = BTreeSet::new();
    let mut total = 0usize;
    for (index, (number, span)) in mentions.iter().enumerate() {
        let end = mentions
            .get(index + 1)
            .map(|(_, next)| next.start())
            .unwrap_or(lower.len());
        let tail = lower[span.end()..end].trim().trim_end_matches(['.', '。']);
        let scope = if index + 1 < mentions.len() {
            tail.strip_suffix("and").ok_or(())?.trim()
        } else {
            tail
        };
        if !(scope.starts_with("in ") || scope.starts_with("at "))
            || scope.split_whitespace().count() < 2
            || scope.contains(',')
            || contains_bounded_ascii_phrase(scope, "or")
            || !scopes.insert(scope.to_string())
        {
            return Err(());
        }
        total = total.checked_add(*number).ok_or(())?;
    }
    Ok(Some((total, format!("{total} times"))))
}

fn count_value_from_claim(value: &str) -> Option<usize> {
    explicit_count_value_with_number(value)
        .map(|(number, _)| number)
        .or_else(|| {
            number_regex()
                .find(value)
                .and_then(|value| value.as_str().parse().ok())
        })
}

fn evidence_ids_for_claim_atom(atoms: &[EvidenceAtom], atom: &EvidenceAtom) -> Vec<u64> {
    let mut evidence_ids = vec![atom.memory_id];
    if !atom
        .supports_slots
        .iter()
        .any(|slot| slot == "evidence_chain_primary")
    {
        return evidence_ids;
    }
    let relation_keys = atom
        .supports_slots
        .iter()
        .filter_map(|slot| slot.strip_prefix("evidence_chain_relation:"))
        .collect::<HashSet<_>>();
    evidence_ids.extend(
        atoms
            .iter()
            .filter(|candidate| {
                candidate
                    .supports_slots
                    .iter()
                    .any(|slot| slot == "evidence_chain_member")
                    && candidate.supports_slots.iter().any(|slot| {
                        slot.strip_prefix("evidence_chain_relation:")
                            .is_some_and(|relation| relation_keys.contains(relation))
                    })
            })
            .map(|candidate| candidate.memory_id),
    );
    evidence_ids.sort_unstable();
    evidence_ids.dedup();
    evidence_ids
}

fn initial_failure_class(
    eligible_count: usize,
    retrieved_count: usize,
    slot_coverage: &[AnswerSlotCoverageReceipt],
    generated: bool,
) -> String {
    if eligible_count == 0 || retrieved_count == 0 {
        "retrieval_miss"
    } else if slot_coverage.iter().any(|slot| slot.status == "missing") {
        "slot_rerank_drop"
    } else if !generated {
        "claim_planning_error"
    } else {
        "none"
    }
    .to_string()
}

fn evidence_ids_for_slot(atoms: &[EvidenceAtom], slot: &str) -> Vec<u64> {
    atoms
        .iter()
        .filter(|atom| atom.supports_slots.iter().any(|value| value == slot))
        .map(|atom| atom.memory_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn known_speakers(items: &[RetrievedItem]) -> Vec<String> {
    let mut speakers = Vec::new();
    for item in items {
        for line in item.content.lines() {
            let (speaker, _) = split_speaker_line(line.trim());
            if let Some(speaker) = speaker {
                speakers.push(speaker);
            }
        }
    }
    let mut seen = HashSet::new();
    speakers.retain(|speaker| {
        let key = normalize_person_key(speaker);
        !key.is_empty() && !is_generic_speaker(&key) && seen.insert(key)
    });
    speakers
}

fn query_mentions_actor(query_lower: &str, candidate: &str) -> bool {
    let candidate_lower = candidate.to_lowercase();
    query_lower
        .match_indices(&candidate_lower)
        .any(|(start, _)| {
            let end = start + candidate_lower.len();
            let before_character = query_lower[..start].chars().next_back();
            let after_character = query_lower[end..].chars().next();
            !before_character.is_some_and(|character| character.is_ascii_alphanumeric())
                && !after_character.is_some_and(|character| character.is_ascii_alphanumeric())
        })
}

fn grammatical_subject_capture(query: &str) -> Option<String> {
    let captures = subject_regex().captures(query)?;
    let whole = captures.get(0)?;
    let candidate = captures.get(1)?;
    let auxiliary = whole
        .as_str()
        .split_whitespace()
        .next()?
        .to_ascii_lowercase();

    if matches!(auxiliary.as_str(), "is" | "are" | "was" | "were") {
        let prefix_terms = word_regex()
            .find_iter(&query[..whole.start()])
            .map(|term| term.as_str().to_ascii_lowercase())
            .collect::<Vec<_>>();
        let fronted_answer_phrase = matches!(
            prefix_terms.first().map(String::as_str),
            Some("what" | "which")
        ) && prefix_terms.len() > 1;
        let suffix_has_term = word_regex().is_match(&query[candidate.end()..]);
        let value = candidate.as_str();
        let lower = value.to_ascii_lowercase();
        let lowercase_predicate_form = value
            .chars()
            .next()
            .is_some_and(char::is_lowercase)
            && (lower.ends_with("ed")
                || lower.ends_with("ing")
                || lower.ends_with("ly"));

        // In a subject question such as "Which deployment region is
        // recorded?", the token after the copula is a predicate, not the
        // subject. The same applies to an explicit participle/modifier before
        // a longer passive predicate. Conversely, an inverted object question
        // ("Which book was deployment reading?") has a subject followed by a
        // predicate and remains admissible. Keeping that distinction here
        // prevents a guessed predicate from becoming a hard fact-subject gate.
        if matches!(
            prefix_terms.first().map(String::as_str),
            Some("what" | "which")
        ) && (lowercase_predicate_form || (fronted_answer_phrase && !suffix_has_term))
        {
            return None;
        }
    }

    let value = candidate.as_str();
    // The subject regex intentionally stops before digits. A hyphenated
    // project/memory identifier such as `capsule-portability-token-6174`
    // would otherwise be truncated to `capsule-portability-token-` and become
    // a false hard subject gate for otherwise relevant evidence.
    if value.ends_with('-') {
        return None;
    }
    Some(strip_person_possessive(value))
}

/// "When Jon has lost his job?" is not inverted: the subject precedes the
/// auxiliary, and the word after it ("lost") is the verb.
fn uninverted_subject_capture(query: &str) -> Option<String> {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX
        .get_or_init(|| {
            Regex::new(r"\b([A-Z][a-z]+)(?:['’]s)?\s+(?:has|have|had|is|was|were|does|did|will|would)\s+[a-z]")
                .expect("uninverted subject regex")
        })
        .captures(query)
        .and_then(|captures| captures.get(1))
        .map(|name| name.as_str().to_string())
        .filter(|name| {
            let lower = name.to_lowercase();
            likely_person_name(name) && !is_question_word(&lower) && !is_temporal_name(&lower)
        })
}

fn discover_subject(query: &str, items: &[RetrievedItem]) -> Option<String> {
    let lower = query.to_lowercase();
    let mut speakers = known_speakers(items);
    let inverted = grammatical_subject_capture(query);
    const RELATION_NOUNS: &[&str] = &[
        "grandma", "grandpa", "grandmother", "grandfather", "mom", "mother", "dad", "father",
        "parents", "son", "daughter", "kids", "children", "brother", "sister", "wife", "husband",
        "partner", "friend", "friends", "aunt", "uncle", "cousin", "family", "boss", "coworker",
    ];
    let lowercase_word = |value: &String| value.chars().next().is_some_and(char::is_lowercase);
    let subject_capture = match (&inverted, uninverted_subject_capture(query)) {
        (Some(verb), Some(name)) if lowercase_word(verb) => Some(name),
        // "What events for veterans has John...", "What made Gina choose ... her
        // store": a lowercase capture that is not a relation noun is not the
        // subject when the question names a participant.
        (Some(word), None)
            if lowercase_word(word)
                && !RELATION_NOUNS.contains(&word.to_lowercase().as_str())
                && speakers.iter().any(|speaker| query_mentions_actor(&lower, speaker)) =>
        {
            None
        }
        _ => inverted,
    };
    if let Some(subject) = subject_capture
        .filter(|value| {
            !is_question_word(&value.to_lowercase()) && !is_temporal_name(&value.to_lowercase())
        })
    {
        return Some(
            speakers
                .iter()
                .find(|speaker| normalize_person_key(speaker) == normalize_person_key(&subject))
                .cloned()
                .unwrap_or(subject),
        );
    }
    speakers.sort_by_key(|value| std::cmp::Reverse(value.chars().count()));
    if let Some(speaker) = speakers
        .into_iter()
        .find(|speaker| query_mentions_actor(&lower, speaker))
    {
        return Some(speaker);
    }

    grammatical_subject_capture(query)
        .filter(|value| {
            let normalized = value.to_lowercase();
            !is_question_word(&normalized) && !is_temporal_name(&normalized)
        })
}

fn discover_people(query: &str, items: &[RetrievedItem]) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut seen = HashSet::new();
    let mut people = known_speakers(items)
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| {
            let normalized = normalize_person_key(value);
            query_mentions_actor(&lower, value) && seen.insert(normalized)
        })
        .collect::<Vec<_>>();
    let grammatical_people = person_name_regex()
        .find_iter(query)
        .map(|capture| capture.as_str().trim_end_matches("'s").to_string())
        .filter(|value| {
            let normalized = value.to_lowercase();
            !is_question_word(&normalized)
                && !is_temporal_name(&normalized)
                && query_mentions_actor(&lower, value)
        })
        .collect::<Vec<_>>();
    let coordinated_subject = grammatical_subject_capture(query)
        .is_some_and(|subject| {
            grammatical_people.iter().any(|person| {
                normalize_person_key(person) == normalize_person_key(&subject)
            })
        });
    let adjacent_subjects = person_name_regex()
        .find_iter(query)
        .collect::<Vec<_>>()
        .windows(2)
        .any(|pair| {
            query[pair[0].end()..pair[1].start()]
                .trim()
                .eq_ignore_ascii_case("and")
                && word_regex()
                    .find(&query[pair[1].end()..])
                    .is_some_and(|word| {
                        matches!(
                            word.as_str().to_lowercase().as_str(),
                            "did"
                                | "do"
                                | "does"
                                | "have"
                                | "has"
                                | "had"
                                | "are"
                                | "were"
                                | "will"
                                | "would"
                        )
                    })
        });
    if grammatical_people.len() >= 2
        && lower.contains(" and ")
        && (coordinated_subject || adjacent_subjects)
    {
        for person in grammatical_people {
            if seen.insert(normalize_person_key(&person)) {
                people.push(person);
            }
        }
    }
    people.sort_by_key(|value| lower.find(&value.to_lowercase()).unwrap_or(usize::MAX));
    people.truncate(8);
    people
}

const NAMED_ENTITY_MODIFIER_SUFFIXES: [&str; 5] =
    ["-related", "-based", "-themed", "-inspired", "-style"];

fn normalize_named_entity_capture(value: &str) -> Option<String> {
    let mut words = value
        .trim_end_matches("'s")
        .trim_end_matches("’s")
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    while words.first().is_some_and(|word| {
        matches!(
            word.to_ascii_lowercase().as_str(),
            "what"
                | "which"
                | "who"
                | "where"
                | "when"
                | "how"
                | "would"
                | "could"
                | "should"
                | "will"
                | "can"
                | "may"
                | "might"
                | "do"
                | "does"
                | "did"
                | "has"
                | "have"
                | "had"
                | "is"
                | "are"
                | "was"
                | "were"
        )
    }) {
        words.remove(0);
    }
    if let Some(last) = words.last_mut() {
        for suffix in NAMED_ENTITY_MODIFIER_SUFFIXES {
            if last.to_ascii_lowercase().ends_with(suffix) {
                last.truncate(last.len() - suffix.len());
                break;
            }
        }
    }
    let normalized = words.join(" ");
    (!normalized.is_empty()).then_some(normalized)
}

fn named_entity_is_request_actor(query: &str, entity: &str) -> bool {
    let query = query.to_ascii_lowercase();
    let entity = entity.to_ascii_lowercase();
    // UI/UX are domain nouns in design questions, not source-side named
    // entities. A capitalized `UI` is otherwise captured by the generic
    // named-entity regex and creates an impossible named_entity:ui slot for
    // typed design-profile summaries whose actual scope is source_id/title.
    // Keep this scoped to design wording so a genuine actor named UI in an
    // unrelated factual query is not silently discarded.
    if matches!(entity.as_str(), "ui" | "ux")
        && contains_any(
            &query,
            &[
                "design",
                "typography",
                "layout",
                "palette",
                "responsive",
                "accessibility",
            ],
        )
    {
        return true;
    }
    // Product/context labels are query scaffolding, not source-side entities.
    // Treat them as request actors only in resume/import wording so a genuine
    // project named “Memory Capsule” in an unrelated fact query remains
    // eligible for normal named-entity matching.
    if entity == "memory capsule"
        && contains_any(&query, &["resume", "import", "context", "capsule"])
    {
        return true;
    }
    // In profile/design questions an entity can be the recipient of the
    // requested adaptation rather than a source-side fact subject (for
    // example, "future UI work" in "what style should future UI work
    // follow?"). Do not force every profile summary to repeat that target.
    if query.contains("should")
        && contains_bounded_ascii_phrase(&query, &format!("{entity} work"))
    {
        return true;
    }
    ["should", "could", "can", "would", "may", "might"]
        .into_iter()
        .any(|modal| {
            ["use", "apply", "follow", "adopt"]
                .into_iter()
                .any(|action| {
                    contains_bounded_ascii_phrase(
                        &query,
                        &format!("{modal} {entity} {action}"),
                    )
                })
        })
}

fn discover_named_entities(
    query: &str,
    people: &[String],
    subject: Option<&str>,
    time_terms: &[String],
) -> Vec<String> {
    let people = people
        .iter()
        .map(|value| normalize_person_key(value))
        .collect::<HashSet<_>>();
    let subject = subject.map(normalize_person_key);
    let temporal = time_terms
        .iter()
        .map(|value| value.to_lowercase())
        .collect::<HashSet<_>>();
    let mut seen = HashSet::new();
    named_entity_sequence_regex()
        .find_iter(query)
        .filter_map(|capture| normalize_named_entity_capture(capture.as_str()))
        .filter(|value| {
            let lower = value.to_lowercase();
            let normalized = normalize_person_key(value);
            !normalized.is_empty()
                && !is_question_word(&lower)
                && !is_temporal_name(&lower)
                && !temporal.contains(&lower)
                && !people.contains(&normalized)
                && subject.as_ref() != Some(&normalized)
                && !named_entity_is_request_actor(query, value)
                && seen.insert(normalized)
        })
        .collect()
}

fn is_temporal_name(value: &str) -> bool {
    matches!(
        value,
        "monday"
            | "tuesday"
            | "wednesday"
            | "thursday"
            | "friday"
            | "saturday"
            | "sunday"
            | "january"
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
}

fn query_requests_temporal_answer(query_lower: &str) -> bool {
    let query_lower = query_lower.trim_start();
    let english = [
        "when",
        "what date",
        "which date",
        "what day",
        "which day",
        "what month",
        "which month",
        "what year",
        "which year",
        "in what year",
        "in which year",
        "in what month",
        "in which month",
    ]
    .iter()
    .any(|prefix| {
        query_lower.strip_prefix(prefix).is_some_and(|rest| {
            // "In which month's game ..." asks for a month too.
            let rest = rest.strip_prefix("'s").or_else(|| rest.strip_prefix("’s")).unwrap_or(rest);
            rest.chars().next().is_none_or(|character| {
                character.is_whitespace() || matches!(character, '?' | '？')
            })
        })
    });
    english || contains_any(query_lower, &["何时", "什么时候", "哪天"])
}

fn asks_for_aspect_set(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "like about",
            "love about",
            "enjoy about",
            "appreciate about",
            "admire about",
            "value about",
            "favorite thing about",
            "favorite things about",
            "what qualities",
            "which qualities",
            "what traits",
            "which traits",
            "what features",
            "which features",
            "what aspects",
            "which aspects",
            "哪些方面",
            "哪些特点",
            "哪些特质",
            "哪些优点",
        ],
    ) || (query_lower.contains("喜欢")
        && contains_any(query_lower, &["什么", "哪些", "哪方面", "方面"]))
}

fn query_requests_achievement_extremum(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "career-high",
            "career high",
            "personal best",
            "personal-best",
            "personal record",
            "record high",
            "record-breaking",
            "record breaking",
            "highest ever",
            "all-time high",
            "职业生涯新高",
            "生涯新高",
            "个人最佳",
            "个人纪录",
            "个人记录",
            "最高纪录",
            "破纪录",
            "刷新纪录",
            "刷新记录",
            "历史最高",
            "最佳成绩",
        ],
    )
}

fn query_requests_media_provenance(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "picture",
            "photo",
            "image",
            "video",
            "audio",
            "recording",
            "照片",
            "图片",
            "视频",
            "音频",
        ],
    ) && contains_any(
        query_lower,
        &[
            "what kind",
            "which picture",
            "which photo",
            "what was the picture",
            "what was the photo",
            "where was",
            "who created",
            "who made",
            "出处",
            "来自哪里",
            "谁创作",
            "什么图片",
        ],
    )
}

fn query_requests_co_participant_event(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "end up",
            "ended up",
            "played",
            "playing",
            "participated",
            "participate",
            "took part",
            "competed",
            "contest",
            "一起参加",
            "一起玩",
            "最终参加",
        ],
    )
}

fn query_requests_preference_recommendation(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "good hobby",
            "what hobby",
            "which hobby",
            "hobby should",
            "hobby would",
            "recommend",
            "suggest",
            "pick up",
            "should try",
            "适合的爱好",
            "推荐什么",
            "建议什么",
        ],
    ) && !asks_what_was_suggested(query_lower)
}

/// "What did Andrew suggest …?", "Which book has Nate recommended?": a
/// question about what someone said, answered from the sources, not a
/// request for a new recommendation.
fn asks_what_was_suggested(query_lower: &str) -> bool {
    static REPORTED: OnceLock<regex::Regex> = OnceLock::new();
    REPORTED.get_or_init(|| regex::Regex::new(
        r"\b(?:did|does|do|has|have|had)\s+(?:[a-z'’]+\s+){1,3}?(?:suggest|recommend)|\b(?:suggested|recommended|reccomended|suggests|recommends)\b|\b(?:suggestions?|recommendations?|advice)\s+(?:did|does|do|has|have|had)\b"
    ).expect("reported suggestion regex")).is_match(query_lower)
}

/// The question asks for a speculative answer (`InferenceType::Likely`).
///
/// `MEMORY_ANSWER_QUESTION_CLASSIFICATION=refined` (default `legacy`):
/// speculative wording ("might", "would", "likely", "potentially", "could",
/// ...) marks a speculative question, while a question about what someone
/// suggested or recommended is a factual question about reported speech;
/// only "recommend"/"suggest" asking for a recommendation stays speculative.
/// Measured on LoCoMo: legacy marks 37 of 96 inference questions and 8 of 446
/// trick questions speculative (all asking what someone suggested), refined
/// 44 and 0. `MEMORY_ANSWER_SPECULATIVE_TERMS` (comma-separated, lower case)
/// replaces the refined term list, e.g. for another language.
fn asks_speculatively(lower: &str) -> bool {
    const LEGACY: &[&str] = &["likely", "probably", "infer", "seem", "could they", "might ", "would ", "recommend", "suggest",
        "推断", "可能", "大概", "看起来", "会不会"];
    const SPECULATIVE: &[&str] = &["likely", "probably", "infer", "seem", "could they", "might ", "would ", "potentially",
        "possibly", "presumably", "perhaps", "could ", "推断", "可能", "大概", "看起来", "会不会"];
    let refined = std::env::var("MEMORY_ANSWER_QUESTION_CLASSIFICATION")
        .is_ok_and(|value| value.trim().eq_ignore_ascii_case("refined"));
    if !refined {
        return contains_any(lower, LEGACY);
    }
    let custom = std::env::var("MEMORY_ANSWER_SPECULATIVE_TERMS").ok().map(|list| {
        list.split(',').map(|term| term.trim().to_lowercase()).filter(|term| !term.is_empty()).collect::<Vec<_>>()
    });
    let speculative = match &custom {
        Some(terms) => terms.iter().any(|term| lower.contains(term.as_str())),
        None => contains_any(lower, SPECULATIVE),
    };
    speculative || ((lower.contains("recommend") || lower.contains("suggest")) && !asks_what_was_suggested(lower))
}

fn is_recommendation_control_term(term: &str) -> bool {
    [
        "a",
        "an",
        "the",
        "his",
        "her",
        "their",
        "my",
        "your",
        "our",
        "for",
        "with",
        "from",
        "into",
        "good",
        "best",
        "hobby",
        "hobbies",
        "related",
        "relation",
        "dream",
        "dreams",
        "would",
        "could",
        "should",
        "pick",
        "up",
        "recommend",
        "suggest",
        "idea",
        "ideas",
        "start",
        "try",
        "new",
    ]
    .contains(&term)
}

fn recommendation_topic_terms(
    query_terms: &[String],
    people: &[String],
    subject: Option<&str>,
    named_entities: &[String],
) -> Vec<String> {
    let excluded = people
        .iter()
        .map(|person| person.to_lowercase())
        .chain(subject.into_iter().map(|value| value.to_lowercase()))
        .chain(named_entities.iter().map(|value| value.to_lowercase()))
        .collect::<HashSet<_>>();
    query_terms
        .iter()
        .filter(|term| {
            !is_recommendation_control_term(term)
                && term.chars().count() >= 3
                && !excluded.contains(&term.to_lowercase())
        })
        .cloned()
        .take(4)
        .collect()
}

fn asks_for_strategy_set(query_lower: &str) -> bool {
    let asks_for_subject_action = [
        "what does ",
        "what do ",
        "how does ",
        "how do ",
        "what can ",
        "what should ",
        "怎么",
        "如何",
    ]
    .iter()
    .any(|prefix| query_lower.trim_start().starts_with(prefix));
    let has_strategy_goal = contains_any(
        query_lower,
        &[
            " to improve",
            " to maintain",
            " to supplement",
            " to prepare",
            " to train",
            " to build",
            " to reduce",
            " to manage",
            " to cope",
            " to balance",
            " to achieve",
            " to support",
            " for training",
            " for recovery",
            " for preparation",
            "提升",
            "训练",
            "准备",
            "恢复",
            "改善",
            "应对",
            "平衡",
        ],
    );
    asks_for_subject_action && has_strategy_goal
}

fn query_requires_complementary_action(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "supplement",
            "complement",
            "alongside",
            "in addition",
            "besides",
            "cross-train",
            "辅助",
            "补充",
            "配合",
            "之外",
        ],
    )
}

fn statement_signals_complementary_action(statement_lower: &str) -> bool {
    contains_any(
        statement_lower,
        &[
            "also",
            "alongside",
            "in addition",
            "extra",
            "additional",
            "both",
            "another",
            "besides",
            "supplement",
            "complement",
            "cross-train",
            "同时",
            "另外",
            "额外",
            "补充",
            "辅助",
            "配合",
        ],
    )
}

fn detect_answer_shape(
    answer_target: AnswerTarget,
    cardinality: Cardinality,
    list_scope: &str,
    inference_type: InferenceType,
    aspect_set_requested: bool,
    strategy_set_requested: bool,
) -> AnswerShape {
    match inference_type {
        InferenceType::Counterfactual => AnswerShape::Counterfactual,
        InferenceType::Likely => AnswerShape::Likely,
        InferenceType::Direct => {
            if aspect_set_requested {
                AnswerShape::AspectSet
            } else if strategy_set_requested {
                AnswerShape::ActionSet
            } else if cardinality == Cardinality::Count {
                AnswerShape::CountAggregate
            } else if cardinality == Cardinality::List {
                if list_scope == "bounded" {
                    AnswerShape::BoundedList
                } else {
                    AnswerShape::List
                }
            } else if answer_target == AnswerTarget::Temporal {
                AnswerShape::Temporal
            } else if answer_target == AnswerTarget::Location {
                AnswerShape::Location
            } else if answer_target == AnswerTarget::Initiation {
                AnswerShape::Initiation
            } else {
                AnswerShape::Scalar
            }
        }
    }
}

fn temporal_week_reference(query_lower: &str) -> bool {
    static ASCII_WEEK: OnceLock<Regex> = OnceLock::new();
    ASCII_WEEK
        .get_or_init(|| Regex::new(r"(?i)\bweek\b").expect("temporal week regex"))
        .is_match(query_lower)
        || query_lower.contains('周')
}

fn extract_temporal_constraint(query: &str) -> TemporalConstraint {
    static RELATIVE_WEEK: OnceLock<Regex> = OnceLock::new();
    static WEEK_OF_MONTH: OnceLock<Regex> = OnceLock::new();
    static MONTH_OF_YEAR: OnceLock<Regex> = OnceLock::new();
    let relative_week = RELATIVE_WEEK.get_or_init(|| {
        Regex::new(r"(?i)\b(?:the\s+)?week\s+(before|after)\b")
            .expect("relative week constraint regex")
    });
    if let Some(captures) = relative_week.captures(query) {
        if let Some(anchor) = parse_explicit_temporal_date(query) {
            return match captures
                .get(1)
                .map(|value| value.as_str().to_ascii_lowercase())
                .as_deref()
            {
                Some("before") => TemporalConstraint::WeekBeforeDate(anchor),
                Some("after") => TemporalConstraint::WeekAfterDate(anchor),
                _ => TemporalConstraint::None,
            };
        }
    }

    let week_of_month = WEEK_OF_MONTH.get_or_init(|| {
        Regex::new(
            r"(?i)\b(first|second|third|fourth|fifth|1st|2nd|3rd|4th|5th)\s+week\s+(?:of|in)\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:,?\s+((?:19|20)\d{2}))?\b",
        )
        .expect("week of month constraint regex")
    });
    if let Some(captures) = week_of_month.captures(query) {
        let Some(week) = captures
            .get(1)
            .and_then(|value| temporal_week_ordinal(value.as_str()))
        else {
            return TemporalConstraint::None;
        };
        let Some(month) = captures
            .get(2)
            .and_then(|value| temporal_month_number(value.as_str()))
        else {
            return TemporalConstraint::None;
        };
        let year = captures
            .get(3)
            .and_then(|value| value.as_str().parse::<i32>().ok());
        return TemporalConstraint::WeekOfMonth { week, month, year };
    }

    let month_of_year = MONTH_OF_YEAR.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:in|during|on)\s+(?:the\s+)?(january|february|march|april|may|june|july|august|september|october|november|december)(?:,?\s+((?:19|20)\d{2}))?\b",
        )
        .expect("month-of-year constraint regex")
    });
    let Some(captures) = month_of_year.captures(query) else {
        return TemporalConstraint::None;
    };
    let Some(month) = captures
        .get(1)
        .and_then(|value| temporal_month_number(value.as_str()))
    else {
        return TemporalConstraint::None;
    };
    let year = captures
        .get(2)
        .and_then(|value| value.as_str().parse::<i32>().ok());
    TemporalConstraint::MonthOfYear { month, year }
}

fn parse_explicit_temporal_date(value: &str) -> Option<chrono::NaiveDate> {
    date_regex().find_iter(value).find_map(|candidate| {
        crate::memory_engine_summary::source_session_date(&format!(
            "Session date: {}",
            candidate.as_str()
        ))
    })
}

fn temporal_week_ordinal(value: &str) -> Option<u8> {
    match value.to_ascii_lowercase().as_str() {
        "first" | "1st" => Some(1),
        "second" | "2nd" => Some(2),
        "third" | "3rd" => Some(3),
        "fourth" | "4th" => Some(4),
        "fifth" | "5th" => Some(5),
        _ => None,
    }
}

fn temporal_month_number(value: &str) -> Option<u32> {
    match value.to_ascii_lowercase().as_str() {
        "january" => Some(1),
        "february" => Some(2),
        "march" => Some(3),
        "april" => Some(4),
        "may" => Some(5),
        "june" => Some(6),
        "july" => Some(7),
        "august" => Some(8),
        "september" => Some(9),
        "october" => Some(10),
        "november" => Some(11),
        "december" => Some(12),
        _ => None,
    }
}

fn temporal_month_name(month: u32) -> &'static str {
    match month {
        1 => "january",
        2 => "february",
        3 => "march",
        4 => "april",
        5 => "may",
        6 => "june",
        7 => "july",
        8 => "august",
        9 => "september",
        10 => "october",
        11 => "november",
        12 => "december",
        _ => "month",
    }
}

fn detect_temporal_granularity(query_lower: &str) -> TemporalGranularity {
    if temporal_week_reference(query_lower) {
        TemporalGranularity::Week
    } else if contains_any(
        query_lower,
        &[
            "what year",
            "which year",
            "the year",
            "哪年",
            "哪一年",
            "几年",
        ],
    ) {
        TemporalGranularity::Year
    } else if contains_any(
        query_lower,
        &[
            "what month",
            "which month",
            "the month",
            "月份",
            "几月",
            "哪个月",
        ],
    ) {
        TemporalGranularity::Month
    } else if contains_any(
        query_lower,
        &["what week", "which week", "the week", "哪周", "第几周"],
    ) {
        TemporalGranularity::Week
    } else if contains_any(
        query_lower,
        &[
            "what date",
            "which date",
            "what day",
            "which day",
            "具体日期",
            "哪天",
            "日期",
        ],
    ) {
        TemporalGranularity::Day
    } else {
        TemporalGranularity::Unspecified
    }
}

pub(crate) fn requested_temporal_granularity(query: &str) -> &'static str {
    detect_temporal_granularity(&query.to_lowercase()).as_str()
}

fn extract_answer_class_terms(
    query_lower: &str,
    cardinality: Cardinality,
    answer_target: AnswerTarget,
) -> Vec<String> {
    if cardinality != Cardinality::List || answer_target != AnswerTarget::List {
        return Vec::new();
    }

    let tokens = word_regex()
        .find_iter(query_lower)
        .map(|value| value.as_str().to_lowercase())
        .collect::<Vec<_>>();
    let Some(auxiliary_index) = tokens.iter().position(|token| {
        matches!(
            token.as_str(),
            "does" | "do" | "did" | "are" | "is" | "were" | "was" | "has" | "have"
        )
    }) else {
        return Vec::new();
    };

    const GENERIC_RESULT_CLASSES: &[&str] = &[
        "things",
        "items",
        "aspects",
        "details",
        "ways",
        "types",
        "kinds",
        "reasons",
        "people",
        "events",
        "activities",
        "projects",
    ];
    tokens[..auxiliary_index]
        .iter()
        .rev()
        .find(|token| {
            token.len() > 3
                && token.ends_with('s')
                && !GENERIC_RESULT_CLASSES.contains(&token.as_str())
        })
        .cloned()
        .into_iter()
        .collect()
}

fn query_requests_initiation_event(query_lower: &str) -> bool {
    let english_prompt = contains_any(
        query_lower,
        &["how did", "how was", "what got", "what introduced"],
    );
    let english_initiation = contains_any(
        query_lower,
        &[
            "get introduced to",
            "got introduced to",
            "introduced to",
            "get into",
            "got into",
            "start ",
            "started ",
            "begin ",
            "began ",
            "take up",
            "took up",
        ],
    );
    let chinese_prompt = contains_any(query_lower, &["如何", "怎么", "怎样"]);
    let chinese_initiation = contains_any(
        query_lower,
        &["开始", "接触", "入门", "入行", "学习", "加入", "报名"],
    );
    (english_prompt && english_initiation) || (chinese_prompt && chinese_initiation)
}

fn statement_signals_initiation_event(statement_lower: &str) -> bool {
    contains_any(
        statement_lower,
        &[
            "introduced me to",
            "introduced us to",
            "introduced him to",
            "introduced her to",
            "introduced them to",
            "signed me up",
            "signed us up",
            "signed him up",
            "signed her up",
            "signed them up",
            "enrolled me",
            "enrolled us",
            "enrolled him",
            "enrolled her",
            "enrolled them",
            "got me into",
            "got us into",
            "got him into",
            "got her into",
            "got them into",
            "first started",
            "started playing",
            "started learning",
            "started training",
            "began playing",
            "began learning",
            "began training",
            "took up",
            "joined a",
            "joined an",
            "joined the",
            "给我报名",
            "带我接触",
            "带我开始",
            "让我接触",
            "开始学习",
            "开始从事",
            "开始练习",
            "开始参加",
            "第一次参加",
            "入门",
        ],
    )
}

fn detect_answer_target(query_lower: &str, cardinality: Cardinality) -> AnswerTarget {
    if crate::memory_engine_summary::temporal_duration_requested(query_lower) {
        AnswerTarget::Duration
    } else if cardinality == Cardinality::Count {
        AnswerTarget::Count
    } else if cardinality == Cardinality::List {
        AnswerTarget::List
    } else if query_requests_temporal_answer(query_lower) {
        AnswerTarget::Temporal
    } else if is_open_event_question(query_lower) {
        AnswerTarget::Event
    } else if contains_any(
        query_lower,
        &[
            "where",
            "which city",
            "what city",
            "which country",
            "what country",
            "location",
            "哪里",
            "哪座城市",
        ],
    ) {
        AnswerTarget::Location
    } else if contains_any(query_lower, &["how long", "duration", "多久", "多长时间"]) {
        AnswerTarget::Duration
    } else if contains_any(query_lower, &["why", "reason", "为什么", "原因"]) {
        AnswerTarget::Reason
    } else if contains_any(
        query_lower,
        &[
            "describe",
            "look like",
            "appearance",
            "change",
            "描述",
            "外观",
            "变化",
        ],
    ) {
        AnswerTarget::Description
    } else if query_requests_initiation_event(query_lower) {
        AnswerTarget::Initiation
    } else if contains_any(query_lower, &["how did", "how does", "如何", "怎么"]) {
        AnswerTarget::Manner
    } else if contains_any(query_lower, &["who", "which person", "谁", "哪位"]) {
        AnswerTarget::Entity
    } else {
        AnswerTarget::Object
    }
}

fn normalize_person_key(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphabetic())
        .flat_map(char::to_lowercase)
        .collect()
}

fn person_slot_name(person: &str) -> String {
    format!("person:{}", normalize_person_key(person))
}

fn named_entity_slot_name(entity: &str) -> String {
    let key = entity
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    format!("named_entity:{key}")
}
fn qualifier_slot_name(term: &str) -> String {
    format!("qualifier:{term}")
}

fn is_generic_speaker(value: &str) -> bool {
    matches!(
        value,
        "user"
            | "assistant"
            | "system"
            | "human"
            | "ai"
            | "operator"
            | "speaker"
            // Structural labels from summary/profile records are not people
            // even when a query repeats the label (for example, "the
            // summary" in a section-summary request).
            | "summary"
            | "evidence"
            | "keyevents"
            | "notes"
            | "signals"
            | "dimension"
            | "memorytype"
            | "source"
            | "content"
            | "title"
            | "profile"
    )
}

fn extract_time_terms(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut terms = Vec::new();
    terms.extend(
        date_regex()
            .find_iter(query)
            .map(|value| value.as_str().to_lowercase()),
    );
    for marker in [
        "today",
        "yesterday",
        "tomorrow",
        "last night",
        "last week",
        "next week",
        "last month",
        "next month",
        "last year",
        "next year",
        "before",
        "after",
        "ago",
        "latest",
        "recent",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "今天",
        "昨天",
        "明天",
        "昨晚",
        "上周",
        "下周",
        "上月",
        "下月",
        "去年",
        "明年",
        "之前",
        "之后",
        "最近",
        "最新",
    ] {
        if (marker.is_ascii() && contains_bounded_ascii_phrase(&lower, marker))
            || (!marker.is_ascii() && lower.contains(marker))
        {
            terms.push(marker.to_string());
        }
    }
    for token in lower.split(|character: char| !character.is_ascii_alphanumeric()) {
        if matches!(
            token,
            "jan"
                | "feb"
                | "mar"
                | "apr"
                | "jun"
                | "jul"
                | "aug"
                | "sep"
                | "sept"
                | "oct"
                | "nov"
                | "dec"
        ) {
            terms.push(token.to_string());
        }
    }

    let has_named_month = terms.iter().any(|term| {
        matches!(
            term.as_str(),
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
                | "jan"
                | "feb"
                | "mar"
                | "apr"
                | "jun"
                | "jul"
                | "aug"
                | "sep"
                | "sept"
                | "oct"
                | "nov"
                | "dec"
        )
    });
    for token in lower.split(|character: char| !character.is_ascii_alphanumeric()) {
        let year = token.len() == 4
            && token
                .parse::<u16>()
                .is_ok_and(|year| (1900..=2100).contains(&year));
        let ordinal_day = token
            .find(|character: char| !character.is_ascii_digit())
            .is_some_and(|split| {
                let (day, suffix) = token.split_at(split);
                has_named_month
                    && matches!(suffix, "st" | "nd" | "rd" | "th")
                    && day.parse::<u8>().is_ok_and(|day| (1..=31).contains(&day))
            });
        if year || ordinal_day {
            terms.push(token.to_string());
        }
    }
    dedup_terms(terms)
}

fn requires_relative_time_reference(time_terms: &[String]) -> bool {
    time_terms
        .iter()
        .any(|term| is_relative_time_reference(term))
}

fn relative_time_reference_matches(time_terms: &[String], value: &str) -> bool {
    let lower = value.to_lowercase();
    time_terms
        .iter()
        .filter(|term| is_relative_time_reference(term))
        .all(|term| {
            (term.is_ascii() && contains_bounded_ascii_phrase(&lower, term))
                || (!term.is_ascii() && lower.contains(term))
        })
}

fn is_relative_time_reference(term: &str) -> bool {
    matches!(
        term,
        "today"
            | "yesterday"
            | "tomorrow"
            | "last night"
            | "last week"
            | "next week"
            | "last month"
            | "next month"
            | "last year"
            | "next year"
            | "今天"
            | "昨天"
            | "明天"
            | "昨晚"
            | "上周"
            | "下周"
            | "上月"
            | "下月"
            | "去年"
            | "明年"
    )
}

fn explicit_quantity_value(value: &str) -> Option<usize> {
    value.parse::<usize>().ok().or_else(|| match value {
        "one" => Some(1),
        "two" | "both" | "couple" | "pair" => Some(2),
        "three" => Some(3),
        "four" => Some(4),
        "five" => Some(5),
        "six" => Some(6),
        "seven" => Some(7),
        "eight" => Some(8),
        "nine" => Some(9),
        "ten" => Some(10),
        "eleven" => Some(11),
        "twelve" => Some(12),
        "thirteen" => Some(13),
        "fourteen" => Some(14),
        "fifteen" => Some(15),
        "sixteen" => Some(16),
        "seventeen" => Some(17),
        "eighteen" => Some(18),
        "nineteen" => Some(19),
        "twenty" => Some(20),
        _ => None,
    })
}

fn extract_quantity_terms(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let mut terms = number_regex()
        .find_iter(query)
        .map(|value| value.as_str().to_string())
        .collect::<Vec<_>>();
    terms.extend(
        word_regex()
            .find_iter(&lower)
            .map(|value| value.as_str().to_string())
            .filter(|value| explicit_quantity_value(value).is_some()),
    );
    for marker in [
        "how many",
        "number of",
        "count",
        "first",
        "last",
        "top",
        "each",
        "every",
        "多少",
        "几次",
        "几个",
        "数量",
        "前",
        "每个",
        "所有",
    ] {
        let present = if marker == "前" {
            lower.match_indices(marker).any(|(start, _)| {
                lower[start + marker.len()..]
                    .trim_start()
                    .chars()
                    .next()
                    .is_some_and(|character| {
                        character.is_ascii_digit()
                            || matches!(
                                character,
                                '一' | '二'
                                    | '两'
                                    | '三'
                                    | '四'
                                    | '五'
                                    | '六'
                                    | '七'
                                    | '八'
                                    | '九'
                                    | '十'
                                    | '百'
                                    | '千'
                                    | '几'
                            )
                    })
            })
        } else {
            (marker.is_ascii() && contains_bounded_ascii_phrase(&lower, marker))
                || (!marker.is_ascii() && lower.contains(marker))
        };
        if present
        {
            terms.push(marker.to_string());
        }
    }
    dedup_terms(terms)
}

fn detect_list_scope(
    query_lower: &str,
    cardinality: Cardinality,
    quantity_terms: &[String],
) -> String {
    if cardinality == Cardinality::Count {
        "count"
    } else if cardinality == Cardinality::Single {
        "single"
    } else if contains_any(
        query_lower,
        &[
            "all ",
            "every ",
            "complete list",
            "exhaustive",
            "所有",
            "全部",
            "完整列出",
            "一个不漏",
        ],
    ) {
        "exhaustive"
    } else if quantity_terms
        .iter()
        .any(|value| explicit_quantity_value(value).is_some())
    {
        "bounded"
    } else {
        "open"
    }
    .to_string()
}

fn extract_negation_terms(query_lower: &str) -> Vec<String> {
    let searchable = strip_quoted_query_segments(query_lower);
    let words = word_regex()
        .find_iter(&searchable)
        .map(|value| value.as_str().to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let mut terms = ["not", "never", "no", "without", "except", "cannot"]
        .into_iter()
        .filter(|marker| words.contains(*marker))
        .map(str::to_string)
        .collect::<Vec<_>>();
    for marker in ["can't", "didn't", "doesn't", "wasn't", "weren't"] {
        if contains_bounded_ascii_phrase(&searchable, marker) {
            terms.push(marker.to_string());
        }
    }
    for marker in ["不是", "没有", "从未", "不能", "不包括", "除外"] {
        if searchable.contains(marker) {
            terms.push(marker.to_string());
        }
    }
    terms.sort();
    terms.dedup();
    terms
}

fn strip_quoted_query_segments(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    let mut quote_end = None::<char>;
    for character in value.chars() {
        if let Some(end) = quote_end {
            if character == end {
                quote_end = None;
            }
            continue;
        }
        match character {
            '"' => quote_end = Some('"'),
            '“' => quote_end = Some('”'),
            '‘' => quote_end = Some('’'),
            _ => output.push(character),
        }
    }
    output
}

fn condition_marker_has_boundaries(value: &str, start: usize, marker_len: usize) -> bool {
    let end = start + marker_len;
    let before = value[..start].chars().next_back();
    let after = value[end..].chars().next();
    before.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '\'')
        && after.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '\'')
}

fn condition_marker_is_embedded_lexeme(value: &str, start: usize, marker: &str) -> bool {
    marker == "当"
        && [
            "当前", "当然", "当地", "当时", "当日", "当天", "当年", "当月", "当下",
            "当中", "当初", "当代", "当场", "当局", "当事", "当今",
        ]
        .iter()
        .any(|word| value[start..].starts_with(word))
}

fn condition_marker_in_text(query: &str) -> Option<(usize, usize, ConditionRelation)> {
    let lower = strip_quoted_query_segments(query).to_lowercase();
    let markers = [
        ("only when", ConditionRelation::OnlyWhen),
        ("only if", ConditionRelation::OnlyWhen),
        ("provided that", ConditionRelation::Assuming),
        ("assuming", ConditionRelation::Assuming),
        ("unless", ConditionRelation::Unless),
        ("if", ConditionRelation::If),
        ("仅当", ConditionRelation::OnlyWhen),
        ("除非", ConditionRelation::Unless),
        ("如果", ConditionRelation::If),
        ("只要", ConditionRelation::If),
        ("当", ConditionRelation::If),
    ];
    let mut first = None;
    for (marker, relation) in markers {
        for (start, _) in lower.match_indices(marker) {
            if condition_marker_is_embedded_lexeme(&lower, start, marker) {
                continue;
            }
            if !condition_marker_has_boundaries(&lower, start, marker.len()) {
                continue;
            }
            if first.is_none_or(|(previous, _, _)| start < previous) {
                first = Some((start, marker.len(), relation));
            }
        }
    }
    first
}

fn extract_condition_relation(query: &str) -> ConditionRelation {
    condition_marker_in_text(query)
        .map(|(_, _, relation)| relation)
        .unwrap_or(ConditionRelation::None)
}

fn extract_condition_terms(query: &str) -> Vec<String> {
    let lower = strip_quoted_query_segments(query).to_lowercase();
    let Some((start, marker_len, _)) = condition_marker_in_text(query) else {
        return Vec::new();
    };
    informative_terms(&lower[start + marker_len..])
        .into_iter()
        .take(12)
        .collect()
}

fn participant_relationship_qualifier_query_terms(query: &str) -> Option<Vec<String>> {
    let lower = query.to_lowercase();
    let clause = [
        " with the person ",
        " with a person ",
        " with someone ",
        " with somebody ",
        " with whom ",
        " with the people ",
    ]
    .iter()
    .filter_map(|marker| {
        lower
            .find(marker)
            .map(|start| &lower[start + marker.len()..])
    })
    .min_by_key(|value| value.len())?;
    let terms = informative_terms(clause);
    let relationship_terms = terms
        .into_iter()
        .filter(|term| relationship_communication_term(term))
        .collect::<Vec<_>>();
    (!relationship_terms.is_empty()).then_some(relationship_terms)
}

fn query_has_collective_participant_reference(query_lower: &str) -> bool {
    [
        " and her family",
        " and his family",
        " and their family",
        " with her family",
        " with his family",
        " with their family",
    ]
    .iter()
    .any(|marker| query_lower.contains(marker))
}

fn relationship_communication_term(term: &str) -> bool {
    lexical_term_forms(term).iter().any(|form| {
        matches!(
            form.as_str(),
            "call"
                | "chat"
                | "communicate"
                | "email"
                | "message"
                | "phone"
                | "skype"
                | "speak"
                | "talk"
                | "text"
                | "video-call"
                | "videocall"
        )
    })
}

fn relational_qualifier_query_terms(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let Some(clause_start) = [
        " during ",
        " while ",
        " after ",
        " before ",
        " following ",
        " once ",
        " when ",
    ]
    .iter()
    .filter_map(|marker| lower.find(marker).map(|start| start + marker.len()))
    .min() else {
        return Vec::new();
    };
    informative_terms(&lower[clause_start..])
}
fn query_action_terms(
    query_terms: &[String],
    people: &[String],
    named_entities: &[String],
    time_terms: &[String],
    quantity_terms: &[String],
    negation_terms: &[String],
) -> Vec<String> {
    let excluded = people
        .iter()
        .map(|value| value.to_lowercase())
        .chain(named_entities.iter().map(|value| value.to_lowercase()))
        .chain(time_terms.iter().cloned())
        .chain(quantity_terms.iter().cloned())
        .chain(negation_terms.iter().cloned())
        .collect::<Vec<_>>();
    query_terms
        .iter()
        .filter(|term| {
            !excluded
                .iter()
                .any(|value| value == *term || value.contains(term.as_str()))
                && !is_action_control_term_or_inflection(term)
                && is_likely_predicate_term(term)
        })
        .take(16)
        .cloned()
        .collect()
}

fn query_object_terms(
    query_terms: &[String],
    people: &[String],
    subject: Option<&str>,
    named_entities: &[String],
    time_terms: &[String],
    quantity_terms: &[String],
    negation_terms: &[String],
    condition_terms: &[String],
    action_terms: &[String],
) -> Vec<String> {
    let excluded = people
        .iter()
        .map(|value| value.to_lowercase())
        .chain(subject.into_iter().map(|value| value.to_lowercase()))
        .chain(named_entities.iter().map(|value| value.to_lowercase()))
        .chain(time_terms.iter().cloned())
        .chain(quantity_terms.iter().cloned())
        .chain(negation_terms.iter().cloned())
        .chain(condition_terms.iter().cloned())
        .chain(action_terms.iter().cloned())
        .collect::<Vec<_>>();
    const CONTROL_TERMS: &[&str] = &[
        "amount",
        "answer",
        "article",
        "articles",
        "aspect",
        "aspects",
        "before",
        "after",
        "activity",
        "activities",
        "book",
        "books",
        "city",
        "country",
        "date",
        "day",
        "describe",
        "described",
        "mention",
        "mentioned",
        "say",
        "said",
        "state",
        "stated",
        "event",
        "events",
        "exercise",
        "exercises",
        "fact",
        "facts",
        "instrument",
        "instruments",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "i",
        "it",
        "its",
        "me",
        "mine",
        "my",
        "our",
        "ours",
        "she",
        "their",
        "theirs",
        "them",
        "they",
        "we",
        "you",
        "your",
        "yours",
        "kind",
        "latest",
        "location",
        "many",
        "month",
        "movie",
        "movies",
        "name",
        "number",
        "place",
        "position",
        "project",
        "projects",
        "reason",
        "recently",
        "thing",
        "things",
        "time",
        "times",
        "type",
        "way",
        "year",
        "years",
        "week",
        "weeks",
        "day",
        "days",
        "month",
        "months",
        "performance",
        "performances",
    ];
    const GRAMMAR_CONTROL_TERMS: &[&str] = &[
        "about",
        "as",
        "at",
        "back",
        "be",
        "being",
        "can",
        "could",
        "did",
        "do",
        "does",
        "during",
        "end",
        "for",
        "from",
        "get",
        "good",
        "got",
        "had",
        "has",
        "have",
        "how",
        "in",
        "may",
        "particularly",
        "might",
        "must",
        "of",
        "on",
        "other",
        "part",
        "person",
        "related",
        "that",
        "the",
        "these",
        "this",
        "those",
        "to",
        "up",
        "what",
        "when",
        "which",
        "while",
        "who",
        "will",
        "with",
        "would",
    ];
    query_terms
        .iter()
        .filter(|term| {
            let normalized_term = term.trim_end_matches("'s").trim_end_matches("’s");
            !excluded
                .iter()
                .any(|value| value.as_str() == normalized_term || value.contains(normalized_term))
                && !CONTROL_TERMS.contains(&normalized_term)
                && !GRAMMAR_CONTROL_TERMS.contains(&normalized_term)
                && !is_action_control_term_or_inflection(normalized_term)
                && !NAMED_ENTITY_MODIFIER_SUFFIXES.iter().any(|suffix| {
                    normalized_term.strip_suffix(suffix).is_some_and(|base| {
                        named_entities.iter().any(|entity| {
                            entity
                                .split_whitespace()
                                .next_back()
                                .is_some_and(|tail| tail.eq_ignore_ascii_case(base))
                        })
                    })
                })
        })
        .take(12)
        .cloned()
        .collect()
}

fn is_action_control_term(value: &str) -> bool {
    matches!(
        value,
        "answer"
            | "date"
            | "day"
            | "direct"
            | "event"
            | "events"
            | "fact"
            | "facts"
            | "information"
            | "list"
            | "memory"
            | "month"
            | "question"
            | "record"
            | "records"
            | "time"
            | "year"
            | "答案"
            | "日期"
            | "时间"
            | "记录"
            | "问题"
            | "列出"
    )
}

fn is_action_control_term_or_inflection(value: &str) -> bool {
    is_action_control_term(value)
        || lexical_term_forms(value)
            .iter()
            .any(|form| is_action_control_term(form))
}

fn is_likely_predicate_term(value: &str) -> bool {
    let forms = lexical_term_forms(value);
    forms.iter().any(|value| {
        matches!(
            value.as_str(),
            "achieve"
                | "attend"
                | "become"
                | "begin"
                | "benefit"
                | "build"
                | "buy"
                | "celebrate"
                | "collaborate"
                | "complete"
                | "create"
                | "discuss"
                | "encounter"
                | "excite"
                | "enjoy"
                | "feel"
                | "finish"
                | "give"
                | "go"
                | "happen"
                | "help"
                | "injure"
                | "introduce"
                | "join"
                | "learn"
                | "leave"
                | "like"
                | "love"
                | "make"
                | "marry"
                | "see"
                | "meet"
                | "organize"
                | "participate"
                | "perform"
                | "pick"
                | "play"
                | "read"
                | "recommend"
                | "recover"
                | "research"
                | "share"
                | "skype"
                | "sign"
                | "start"
                | "study"
                | "supplement"
                | "surf"
                | "take"
                | "talk"
                | "travel"
                | "visit"
                | "watch"
                | "win"
                | "write"
        )
    })
}

fn protected_query_terms(
    people: &[String],
    subject: Option<&str>,
    named_entities: &[String],
    time_terms: &[String],
    quantity_terms: &[String],
    negation_terms: &[String],
    condition_terms: &[String],
    object_terms: &[String],
    qualifier_terms: &[String],
) -> Vec<String> {
    dedup_terms(
        people
            .iter()
            .map(|value| value.to_lowercase())
            .chain(subject.into_iter().map(|value| value.to_lowercase()))
            .chain(named_entities.iter().map(|value| value.to_lowercase()))
            .chain(time_terms.iter().cloned())
            .chain(quantity_terms.iter().cloned())
            .chain(negation_terms.iter().cloned())
            .chain(condition_terms.iter().cloned())
            .chain(object_terms.iter().cloned())
            .chain(qualifier_terms.iter().cloned())
            .collect(),
    )
}

fn dedup_terms(values: Vec<String>) -> Vec<String> {
    let mut seen = HashSet::new();
    values
        .into_iter()
        .map(|value| value.trim().to_lowercase())
        .filter(|value| !value.is_empty() && seen.insert(value.clone()))
        .collect()
}

fn lexical_term_forms(term: &str) -> Vec<String> {
    let term = term.to_lowercase();
    let mut forms = vec![term.clone()];
    let irregular_forms: &[&str] = match term.as_str() {
        "go" => &["went", "gone"],
        "see" => &["saw", "seen"],
        "saw" | "seen" => &["see"],
        "went" | "gone" => &["go"],
        "give" => &["gave", "given"],
        "gave" | "given" => &["give"],
        "meet" => &["met"],
        "met" => &["meet"],
        "marry" => &["married"],
        "married" => &["marry"],
        "make" => &["made"],
        "made" => &["make"],
        "take" => &["took", "taken"],
        "took" | "taken" => &["take"],
        "build" => &["built"],
        "built" => &["build"],
        "write" => &["wrote", "written"],
        "writing" => &["write", "wrote", "written"],
        "wrote" | "written" => &["write"],
        "think" => &["thought"],
        "thought" => &["think"],
        _ => &[],
    };
    forms.extend(irregular_forms.iter().map(|form| (*form).to_string()));
    if term.len() > 5 && term.ends_with("ing") {
        forms.push(term[..term.len() - 3].to_string());
    }
    if term.len() > 4 && term.ends_with("ied") {
        let root = &term[..term.len() - 3];
        forms.push(format!("{root}y"));
        forms.push(format!("{root}ying"));
    }
    if term.len() > 4 && term.ends_with("ed") {
        let root = &term[..term.len() - 2];
        forms.push(root.to_string());
        forms.push(format!("{root}e"));
        // Queries commonly use a past-tense verb while the source describes
        // an ongoing action. Keep this lexical, rather than inventing a
        // domain-specific action synonym (camped -> camping, moved -> moving).
        forms.push(format!("{root}ing"));
    }
    if term.len() > 3 && term.ends_with('s') && !term.ends_with("ss") {
        forms.push(term[..term.len() - 1].to_string());
    }
    if term.len() > 3
        && term
            .chars()
            .all(|character| character.is_ascii_alphabetic())
    {
        if term.ends_with('e') {
            forms.push(format!("{term}d"));
            forms.push(format!("{}ing", &term[..term.len() - 1]));
            forms.push(format!("{term}s"));
        } else if term.ends_with('y')
            && term
                .chars()
                .rev()
                .nth(1)
                .is_some_and(|character| !matches!(character, 'a' | 'e' | 'i' | 'o' | 'u'))
        {
            let root = &term[..term.len() - 1];
            forms.push(format!("{root}ied"));
            forms.push(format!("{root}ies"));
            forms.push(format!("{term}ing"));
        } else {
            forms.push(format!("{term}ed"));
            forms.push(format!("{term}ing"));
            forms.push(format!("{term}s"));
        }
    }
    dedup_terms(forms)
}

fn term_forms(term: &str) -> Vec<String> {
    let mut forms = lexical_term_forms(term);
    const CONCEPT_GROUPS: &[&[&str]] = &[
        &[
            "like",
            "love",
            "enjoy",
            "admire",
            "favorite",
            "favourite",
            "hooked",
            "fan",
        ],
        &[
            "travel", "trip", "visit", "go", "went", "gone", "leave", "left", "depart", "departed",
        ],
        &["injury", "injure", "injured", "hurt"],
        &["marry", "married", "marriage", "wedding"],
        &[
            "read", "reading", "reread", "book", "novel", "finish", "finished",
        ],
        &["start", "started", "begin", "began"],
        &[
            "join",
            "joined",
            "enroll",
            "enrolled",
            "sign up",
            "signed up",
        ],
        &["sign", "signed", "contract", "contracted"],
        &[
            "achieve",
            "achieved",
            "achieving",
            "career-high",
            "accomplishment",
            "milestone",
        ],
        &[
            "share",
            "shared",
            "tell",
            "told",
            "talk",
            "talked",
            "discuss",
            "discussed",
        ],
        &["give", "gave", "given", "gift", "receive", "received"],
        &[
            "write", "wrote", "written", "writing", "article", "articles",
        ],
        &[
            "celebrate",
            "celebrated",
            "win",
            "winning",
            "won",
            "victory",
        ],
        &["deal", "deals", "sponsorship", "endorsement"],
        &["exercise", "training", "train", "workout", "yoga"],
        &["challenge", "difficulty", "struggle", "tough"],
        &["describe", "described", "call", "called", "say", "said"],
        &["excited", "excite", "stoked", "thrilled", "pumped", "psyched", "eager"],
        &[
            "play",
            "playing",
            "perform",
            "performance",
            "learning",
            "learn",
        ],
    ];
    for group in CONCEPT_GROUPS {
        if group
            .iter()
            .any(|candidate| forms.iter().any(|form| form == candidate))
        {
            forms.extend(group.iter().map(|value| (*value).to_string()));
        }
    }
    dedup_terms(forms)
}

fn ascii_term_word_byte(value: u8) -> bool {
    value.is_ascii_alphanumeric() || matches!(value, b'\'' | b'_')
}

fn text_matches_term_form(text_lower: &str, form: &str) -> bool {
    let bounded_ascii = !form.is_empty()
        && form
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "'_-".contains(character));
    if !bounded_ascii {
        return text_lower.contains(form);
    }

    let bytes = text_lower.as_bytes();
    for (start, _) in text_lower.match_indices(form) {
        let before_boundary = start == 0 || !ascii_term_word_byte(bytes[start - 1]);
        let end = start + form.len();
        let after_boundary = end == bytes.len() || !ascii_term_word_byte(bytes[end]);
        let possessive_boundary = text_lower
            .get(end..)
            .and_then(|rest| rest.strip_prefix("'s"))
            .is_some_and(|rest| {
                rest.is_empty()
                    || rest
                        .as_bytes()
                        .first()
                        .is_some_and(|value| !ascii_term_word_byte(*value))
            });
        if before_boundary && (after_boundary || possessive_boundary) {
            return true;
        }
    }
    false
}

fn text_matches_named_entity(text_lower: &str, entity: &str) -> bool {
    let entity_lower = entity.to_lowercase();
    if text_matches_term(text_lower, &entity_lower) {
        return true;
    }

    // Conversations commonly shorten a multi-word proper name to its first
    // name or surname. Keep that bridge bounded to the outer name parts;
    // arbitrary component matching would turn an unrelated middle word into
    // entity evidence.
    let name_parts = entity_lower
        .split_whitespace()
        .filter(|part| part.chars().count() >= 3)
        .collect::<Vec<_>>();
    if name_parts.len() < 2 {
        return false;
    }
    let matches_alias = [name_parts[0], name_parts[name_parts.len() - 1]]
        .into_iter()
        .any(|alias| text_matches_term(text_lower, alias));
    matches_alias
}

pub(crate) fn text_matches_term(text_lower: &str, term: &str) -> bool {
    term_forms(term)
        .iter()
        .any(|form| text_matches_term_form(text_lower, form))
}

fn text_matches_lexical_term(text_lower: &str, term: &str) -> bool {
    lexical_term_forms(term)
        .iter()
        .any(|form| text_matches_term_form(text_lower, form))
}

fn matching_term_count(text_lower: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| text_matches_term(text_lower, term))
        .count()
}

fn matching_lexical_term_count(text_lower: &str, terms: &[String]) -> usize {
    terms
        .iter()
        .filter(|term| text_matches_lexical_term(text_lower, term))
        .count()
}

fn required_action_overlap(term_count: usize) -> usize {
    usize::from(term_count > 0)
}

fn required_object_overlap(term_count: usize) -> usize {
    usize::from(term_count > 0)
}





fn requested_modalities(query_lower: &str) -> Vec<String> {
    let mut modalities = vec!["text".to_string()];
    for (modality, terms) in [
        (
            "image",
            &["image", "photo", "picture", "图片", "照片", "图像"][..],
        ),
        ("video", &["video", "footage", "视频", "录像"][..]),
        ("audio", &["audio", "sound", "音频", "语音", "声音"][..]),
        (
            "spatial",
            &["3d", "point cloud", "pose", "空间", "点云", "姿态"][..],
        ),
        ("sensor", &["sensor", "telemetry", "传感器", "遥测"][..]),
    ] {
        if contains_any(query_lower, terms) {
            modalities.push(modality.to_string());
        }
    }
    let stylistic_voice = contains_any(
        query_lower,
        &[
            "writing style",
            "style profile",
            "author style",
            "diction",
            "rhythm",
            "tone",
            "pacing",
            "文风",
            "写作风格",
        ],
    );
    if !stylistic_voice && contains_bounded_ascii_phrase(query_lower, "voice") {
        modalities.push("audio".to_string());
    }
    dedup_terms(modalities)
}

fn informative_terms(value: &str) -> Vec<String> {
    let mut terms = BTreeSet::new();
    for matched in word_regex().find_iter(value) {
        let token = matched.as_str().to_lowercase();
        if token.chars().any(is_cjk) {
            // Bigram each contiguous CJK run separately. Joining across an
            // embedded digit or Latin span (`第1章` -> `第章`) yields a term the
            // source text can never contain.
            let chars = token.chars().collect::<Vec<_>>();
            for run in chars.split(|ch| !is_cjk(*ch)) {
                for window in run.windows(2) {
                    let term = window.iter().collect::<String>();
                    if !is_question_word(&term) {
                        terms.insert(term);
                    }
                }
            }
        } else if token.len() > 1 && !is_question_word(&token) {
            terms.insert(token);
        }
    }
    terms.into_iter().collect()
}

/// Removes request-language residue from Question Plan terms without changing the
/// generic term extractor used to project and retrieve source evidence.
fn question_informative_terms(value: &str) -> Vec<String> {
    let lower = value.to_lowercase();
    let structural_selector_requested = !extract_structural_selectors(value).is_empty();
    // Scope identifiers and a standalone output-format clause are not content
    // requirements. Mask them before CJK bigram tokenization, which otherwise
    // creates boundary fragments such as 第一/一章/给出/出章.
    let content_query = if structural_selector_requested {
        static SCOPE: OnceLock<Regex> = OnceLock::new();
        static FORMAT: OnceLock<Regex> = OnceLock::new();
        static SCOPE_COMPARISON: OnceLock<Regex> = OnceLock::new();
        static SCOPE_OPEN_TAIL: OnceLock<Regex> = OnceLock::new();
        let scope = SCOPE.get_or_init(|| Regex::new(
            r"第\s*(?:[0-9]+|[零〇一二两三四五六七八九十百千]{1,12})(?:\s*[、,，和及与]\s*(?:[0-9]+|[零〇一二两三四五六七八九十百千]{1,12}))*\s*(?:章|节|条|页|步|部|集|幕|回|篇|段|项|款)"
        ).expect("scope token masking"));
        let format = FORMAT.get_or_init(|| Regex::new(
            r"(?:^|[。？！!?；;])\s*请?(?:给出|提供|列出)(?:章节概要|概要|摘要|关键事件|承接前文|故事线|证据|引用|来源|[、，,和及与\s])+(?:[。？！!?；;]|$)"
        ).expect("standalone format instruction"));
        let without_format = format.replace_all(value, " ");
        let comparison = SCOPE_COMPARISON.get_or_init(|| {
            Regex::new(r"^\s*请?(?:比较|对比)(?:所有|全部|各个)(?:文档|文件|小说|作品)(?:的)?\s*$")
                .expect("structural comparison operation")
        });
        // Parse a complete structural operation, including explicit ranges or
        // lists. Any person/action/condition between or after scopes prevents
        // stripping; those remain substantive question constraints.
        let selectors = scope.find_iter(&without_format).collect::<Vec<_>>();
        let open_tail = SCOPE_OPEN_TAIL.get_or_init(|| Regex::new(
            r"^\s*(?:(?:讲了|发生了|发生|讲述了|讲述)(?:什么))?\s*[。？！!?；;.]*\s*$"
        ).expect("open structural question tail"));
        let pure_structural_comparison = selectors.first().zip(selectors.last()).is_some_and(|(first, last)| {
            comparison.is_match(&without_format[..first.start()])
                && open_tail.is_match(&without_format[last.end()..])
                && selectors.windows(2).all(|pair| matches!(
                    without_format[pair[0].end()..pair[1].start()].trim(),
                    "到" | "至" | "-" | "—" | "~" | "～" | "、" | "," | "，" | "和" | "及" | "与"
                ))
        });
        if pure_structural_comparison {
            String::new()
        } else {
            scope.replace_all(&without_format, " ").into_owned()
        }
    } else {
        value.to_string()
    };
    let mut terms = informative_terms(&content_query);
    if is_open_event_question(&lower) {
        let prompt_fragments = open_event_prompt_fragments(&content_query.to_lowercase());
        const ASCII_EVENT_PROMPT_TERMS: &[&str] = &[
            "describe",
            "explain",
            "happen",
            "happened",
            "happening",
            "occur",
            "occurred",
            "occurring",
            "recap",
            "summarize",
            "summary",
            "tell",
        ];
        terms.retain(|term| {
            !prompt_fragments.contains(term)
                && !ASCII_EVENT_PROMPT_TERMS.contains(&term.as_str())
                && !is_structural_selector_fragment(term)
        });
    }
    if structural_selector_requested {
        terms.retain(|term| {
            !is_structural_selector_fragment(term) && !is_structural_selector_keyword(term)
        });
    }
    terms
}

fn is_open_event_question(query_lower: &str) -> bool {
    contains_any(
        query_lower,
        &[
            "what happened",
            "what happened next",
            "what is happening",
            "what was happening",
            "tell me what happened",
            "发生了什么",
            "发生什么",
            "发生了哪些事",
            "发生哪些事",
            "出了什么事",
            "出了什么问题",
            "怎么了",
            "讲了什么",
            "说了什么",
            "写了什么",
            "内容是什么",
        ],
    )
}

/// `informative_terms` indexes CJK text as adjacent-character terms. When an
/// open-event prompt is attached to a selector or noun, remove every adjacent
/// term that overlaps the prompt, including the boundary term (for example,
/// `章发` in `第315章发生了什么`).
fn open_event_prompt_fragments(query_lower: &str) -> BTreeSet<String> {
    const CJK_OPEN_EVENT_MARKERS: &[&str] = &[
        "发生了什么",
        "发生什么",
        "发生了哪些事",
        "发生哪些事",
        "出了什么事",
        "出了什么问题",
        "怎么了",
        "讲了什么",
        "说了什么",
        "写了什么",
        "内容是什么",
    ];

    let mut fragments = BTreeSet::new();
    for matched in word_regex().find_iter(query_lower) {
        let token = matched.as_str();
        if !token.chars().any(is_cjk) {
            continue;
        }
        let marker_ranges = CJK_OPEN_EVENT_MARKERS
            .iter()
            .flat_map(|marker| {
                token
                    .match_indices(marker)
                    .map(move |(start, _)| (start, start + marker.len()))
            })
            .collect::<Vec<_>>();
        if marker_ranges.is_empty() {
            continue;
        }

        let cjk_chars = token
            .char_indices()
            .filter(|(_, character)| is_cjk(*character))
            .collect::<Vec<_>>();
        for window in cjk_chars.windows(2) {
            let (first_start, first) = window[0];
            let (second_start, second) = window[1];
            let pair_end = second_start + second.len_utf8();
            if marker_ranges.iter().any(|(marker_start, marker_end)| {
                first_start < *marker_end && pair_end > *marker_start
            }) {
                fragments.insert(format!("{first}{second}"));
            }
        }
    }
    fragments
}

fn is_structural_selector_fragment(value: &str) -> bool {
    matches!(
        value,
        "第章"
            | "第节"
            | "第条"
            | "第页"
            | "第步"
            | "第部"
            | "第集"
            | "第幕"
            | "第回"
            | "第篇"
            | "第段"
            | "第项"
            | "第款"
    )
}

fn is_structural_selector_keyword(value: &str) -> bool {
    let singular = value.strip_suffix('s').unwrap_or(value);
    matches!(
        singular,
        "chapter"
            | "section"
            | "article"
            | "page"
            | "step"
            | "part"
            | "episode"
            | "scene"
            | "act"
            | "clause"
            | "章节"
            | "页码"
            | "步骤"
    )
}

/// Extracts explicit section-like scopes independently from answer quantities.
fn extract_structural_selectors(value: &str) -> Vec<String> {
    static CJK_SELECTOR: OnceLock<Regex> = OnceLock::new();
    static CJK_SELECTOR_LIST: OnceLock<Regex> = OnceLock::new();
    static ASCII_SELECTOR: OnceLock<Regex> = OnceLock::new();
    static ASCII_SELECTOR_LIST: OnceLock<Regex> = OnceLock::new();
    let cjk_selector = CJK_SELECTOR.get_or_init(|| {
        Regex::new(r"第\s*([0-9]+|[零〇一二两三四五六七八九十百千]{1,12})\s*(章|节|条|页|步|部|集|幕|回|篇|段|项|款)")
            .expect("CJK structural selector regex")
    });
    let cjk_selector_list = CJK_SELECTOR_LIST.get_or_init(|| {
        Regex::new(r"第\s*((?:[0-9]+|[零〇一二两三四五六七八九十百千]{1,12})(?:\s*[、,，和及与]\s*(?:[0-9]+|[零〇一二两三四五六七八九十百千]{1,12}))+?)\s*(章|节|条|页|步|部|集|幕|回|篇|段|项|款)")
            .expect("CJK structural selector list regex")
    });
    let ascii_selector = ASCII_SELECTOR.get_or_init(|| {
        Regex::new(
            r"(?i)\b(chapter|section|article|page|step|part|episode|scene|act|clause)\s*([0-9]+)\b",
        )
        .expect("ASCII structural selector regex")
    });

    let ascii_selector_list = ASCII_SELECTOR_LIST.get_or_init(|| {
        Regex::new(
            r"(?i)\b(chapters|sections|articles|pages|steps|parts|episodes|scenes|acts|clauses)\s+([0-9]+(?:\s*(?:,|and|or|&)\s*(?:and\s+)?[0-9]+)*)\b",
        )
        .expect("ASCII structural selector list regex")
    });
    let mut selectors = BTreeSet::new();
    let cjk_scopes = cjk_selector.captures_iter(value).collect::<Vec<_>>();
    for pair in cjk_scopes.windows(2) {
        let left = pair[0].get(0).unwrap();
        let right = pair[1].get(0).unwrap();
        if !matches!(value[left.end()..right.start()].trim(), "到" | "至" | "-" | "—" | "~" | "～") { continue; }
        if let (Some(a), Some(b)) = (
            canonical_structural_selector(&pair[0][2], &pair[0][1]),
            canonical_structural_selector(&pair[1][2], &pair[1][1]),
        ) {
            let (kind_a, start) = a.split_once(':').unwrap();
            let (kind_b, end) = b.split_once(':').unwrap();
            if kind_a == kind_b && start.parse::<u64>().unwrap() <= end.parse::<u64>().unwrap() {
                selectors.insert(format!("{kind_a}:{start}..{end}"));
            }
        }
    }
    for captures in cjk_scopes {
        let Some(number) = captures.get(1) else {
            continue;
        };
        let Some(kind) = captures.get(2) else {
            continue;
        };
        if let Some(selector) = canonical_structural_selector(kind.as_str(), number.as_str()) {
            selectors.insert(selector);
        }
    }
    for captures in cjk_selector_list.captures_iter(value) {
        let Some(kind) = captures.get(2) else {
            continue;
        };
        let Some(numbers) = captures.get(1) else {
            continue;
        };
        for number in numbers.as_str().split(['、', ',', '，', '和', '及', '与']) {
            let number = number.trim();
            if number.is_empty() {
                continue;
            }
            if let Some(selector) = canonical_structural_selector(kind.as_str(), number) {
                selectors.insert(selector);
            }
        }
    }
    for captures in ascii_selector.captures_iter(value) {
        let Some(kind) = captures.get(1) else {
            continue;
        };
        let Some(number) = captures.get(2) else {
            continue;
        };
        if let Some(selector) = canonical_structural_selector(kind.as_str(), number.as_str()) {
            selectors.insert(selector);
        }
    }
    for captures in ascii_selector_list.captures_iter(value) {
        let Some(kind) = captures.get(1) else {
            continue;
        };
        let Some(numbers) = captures.get(2) else {
            continue;
        };
        for number in numbers
            .as_str()
            .split(|character: char| !character.is_ascii_digit())
        {
            if number.is_empty() {
                continue;
            }
            if let Some(selector) = canonical_structural_selector(kind.as_str(), number) {
                selectors.insert(selector);
            }
        }
    }
    selectors.into_iter().collect()
}

fn canonical_structural_selector(kind: &str, number: &str) -> Option<String> {
    let normalized_kind = kind.to_ascii_lowercase();
    let kind = normalized_kind
        .strip_suffix('s')
        .unwrap_or(&normalized_kind);
    let kind = match kind {
        "章" | "回" | "chapter" => "chapter",
        "节" | "section" => "section",
        "条" | "篇" | "article" => "article",
        "页" | "page" => "page",
        "步" | "step" => "step",
        "部" | "part" => "part",
        "集" | "episode" => "episode",
        "幕" | "scene" | "act" => "scene",
        "段" => "paragraph",
        "项" => "item",
        "款" | "clause" => "clause",
        _ => return None,
    };
    let number = number.parse::<u64>().ok().or_else(|| {
        (number.chars().count() <= 12)
            .then(|| crate::memory_engine_summary::parse_chinese_chapter_number(number))
            .flatten()
            .map(u64::from)
    })?;
    Some(format!("{kind}:{number}"))
}

fn structural_selector_range(selector: &str) -> Option<(&str, u64, u64)> {
    let (kind, bounds) = selector.split_once(':')?;
    let (start, end) = bounds.split_once("..")?;
    let (start, end) = (start.parse().ok()?, end.parse().ok()?);
    (start <= end).then_some((kind, start, end))
}

fn structural_selector_contains(requested: &str, present: &str) -> bool {
    if requested == present { return true; }
    let Some((kind, start, end)) = structural_selector_range(requested) else { return false; };
    let Some((actual_kind, number)) = present.split_once(':') else { return false; };
    actual_kind == kind && number.parse::<u64>().is_ok_and(|number| (start..=end).contains(&number))
}

fn structural_range_fully_observed(selector: &str, atoms: &[&EvidenceAtom]) -> bool {
    let Some((kind, start, end)) = structural_selector_range(selector) else { return true; };
    let numbers = atoms.iter().flat_map(|atom| atom.section_scope.iter())
        .filter_map(|scope| scope.split_once(':'))
        .filter(|(actual_kind, _)| *actual_kind == kind)
        .filter_map(|(_, number)| number.parse::<u64>().ok())
        .filter(|number| (start..=end).contains(number)).collect::<BTreeSet<_>>();
    numbers.len() as u128 == u128::from(end) - u128::from(start) + 1
}

fn matching_structural_selectors(value: &str, selectors: &[String]) -> Vec<String> {
    let present = extract_structural_selectors(value)
        .into_iter()
        .collect::<HashSet<_>>();
    selectors
        .iter()
        .filter(|selector| present.iter().any(|scope| structural_selector_contains(selector, scope)))
        .cloned()
        .collect()
}

fn structural_selector_slot_name(selector: &str) -> String {
    format!("structural_selector:{selector}")
}

fn split_speaker_line(line: &str) -> (Option<String>, &str) {
    let Some(capture) = speaker_line_regex().captures(line) else {
        return (None, line);
    };
    let label = capture
        .get(1)
        .map(|value| value.as_str())
        .unwrap_or_default();
    if !extract_structural_selectors(label).is_empty() {
        // A document heading such as `Section 7:` or `第315章：` is part of
        // the evidence scope, not a dialogue speaker. Keep it on the atom.
        return (None, line);
    }
    let speaker = capture
        .get(1)
        .map(|value| value.as_str().trim().to_string());
    let body = capture.get(2).map(|value| value.as_str()).unwrap_or(line);
    (speaker, body)
}

fn dialogue_turn_source_anchor(line: &str) -> Option<AnswerEvidenceSourceAnchor> {
    let value = dialogue_turn_prefix_regex()
        .captures(line.trim())?
        .get(1)?
        .as_str()
        .to_string();
    Some(AnswerEvidenceSourceAnchor {
        anchor_format: "answer_evidence_source_anchor_v1".to_string(),
        kind: "dialogue_turn".to_string(),
        value,
    })
}

fn split_statements(value: &str) -> Vec<&str> {
    let mut statements = Vec::new();
    let mut start = 0;
    for (offset, character) in value.char_indices() {
        if !matches!(character, '.' | '?' | '!' | '。' | '？' | '！' | ';' | '；') {
            continue;
        }
        if character == '.' {
            let before = &value[..offset];
            let after = &value[offset + 1..];
            let left = before.chars().next_back();
            let right = after.chars().next();
            let token = before
                .rsplit(|ch: char| !ch.is_alphabetic())
                .next()
                .unwrap_or_default();
            if left.is_some_and(|ch| ch.is_ascii_digit())
                && right.is_some_and(|ch| ch.is_ascii_digit())
                || matches!(
                    token.to_ascii_lowercase().as_str(),
                    "dr" | "mr"
                        | "mrs"
                        | "ms"
                        | "prof"
                        | "st"
                        | "sr"
                        | "jr"
                        | "vs"
                        | "e"
                        | "g"
                        | "i"
                )
                || token.len() == 1
                    && token.chars().all(|ch| ch.is_ascii_uppercase())
                    && !before[..before.len() - token.len()]
                        .trim_end()
                        .chars()
                        .next_back()
                        .is_some_and(|ch| ch.is_ascii_digit())
                || left.is_some_and(|ch| ch.is_alphanumeric())
                    && right.is_some_and(|ch| ch.is_alphanumeric())
            {
                continue;
            }
        }
        let end = offset + character.len_utf8();
        if !value[start..end].trim().is_empty() {
            statements.push(value[start..end].trim());
        }
        start = end;
    }
    if !value[start..].trim().is_empty() {
        statements.push(value[start..].trim());
    }
    if statements.is_empty() && !value.trim().is_empty() {
        statements.push(value.trim());
    }
    statements
}

fn clean_answer_statement(value: &str) -> String {
    value
        .trim()
        .trim_start_matches("event:")
        .trim()
        .trim_end_matches(|character: char| character.is_whitespace())
        .to_string()
}

fn render_answer_statement(plan: &QueryPlan, atom: &EvidenceAtom) -> String {
    let statement = clean_answer_statement(&atom.text);
    if plan.co_participant_event_requested {
        if let Some(activity) = shared_event_activity_value(&statement) {
            return activity;
        }
    }
    let Some(subject) = plan.subject.as_deref() else {
        return statement;
    };
    let speaker_matches_subject = atom
        .speaker
        .as_deref()
        .is_some_and(|speaker| normalize_person_key(speaker) == normalize_person_key(subject));
    if plan.answer_target != AnswerTarget::Initiation
        || query_requires_subject_speaker(plan)
        || !speaker_matches_subject
        || !first_person_initiation_perspective_safe(&statement)
    {
        return statement;
    }

    let possessive = if subject.ends_with('s') || subject.ends_with('S') {
        format!("{subject}'")
    } else {
        format!("{subject}'s")
    };
    first_person_token_regex()
        .replace_all(&statement, |captures: &regex::Captures<'_>| match captures
            .get(0)
            .map(|capture| capture.as_str().to_ascii_lowercase())
            .as_deref()
        {
            Some("i'm") => format!("{subject} is"),
            Some("i've") => format!("{subject} has"),
            Some("i'll") => format!("{subject} will"),
            Some("i'd") => format!("{subject} would"),
            Some("my") | Some("mine") => possessive.clone(),
            Some("me") | Some("i") => subject.to_string(),
            _ => captures
                .get(0)
                .map(|capture| capture.as_str().to_string())
                .unwrap_or_default(),
        })
        .into_owned()
}

fn first_person_initiation_perspective_safe(value: &str) -> bool {
    let lower = value.to_lowercase();
    contains_any(
        &lower,
        &[
            "i was ",
            "i had ",
            "i got ",
            "i started ",
            "i began ",
            "i joined ",
            "i took ",
            "i learned ",
            "i enrolled ",
            "i turned ",
            "i moved ",
            "i met ",
            "i signed ",
            "i watched ",
            "i played ",
            "i trained ",
            "i studied ",
            "i became ",
            "i've ",
            "i'd ",
            "i'll ",
            "i'm ",
            "signed me up",
            "enrolled me",
        ],
    )
}

fn first_person_token_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)\b(?:i'm|i've|i'll|i'd|i|me|my|mine)\b")
            .expect("first-person answer perspective regex")
    })
}

fn normalize_for_dedup(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn is_session_date_line(lower: &str) -> bool {
    lower.starts_with("session date:")
        || lower.starts_with("conversation date:")
        || lower.starts_with("会话日期：")
        || lower.starts_with("会话日期:")
}

fn is_system_metadata_line(lower: &str) -> bool {
    contains_any(
        lower,
        &[
            "conversation id:",
            "session id:",
            "source id:",
            "document id:",
            "memory type:",
            "系统元数据：",
        ],
    )
}

pub(crate) fn contains_date_like_value(value: &str) -> bool {
    date_regex().is_match(value)
}

pub(crate) fn contains_bare_calendar_month_reference(value: &str) -> bool {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)\b(?:in|during|by|until|from|through|this|next|last|early|late)\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b")
            .expect("answer evidence calendar month regex")
    })
    .is_match(value)
}

fn source_session_month_year(value: &str) -> Option<String> {
    static ENGLISH_MONTH_YEAR: OnceLock<Regex> = OnceLock::new();
    static CHINESE_MONTH_YEAR: OnceLock<Regex> = OnceLock::new();
    let english = ENGLISH_MONTH_YEAR.get_or_init(|| {
        Regex::new(r"(?i)^(?:session|conversation) date:\s*((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4})\b")
            .expect("session month-year regex")
    });
    let chinese = CHINESE_MONTH_YEAR.get_or_init(|| {
        Regex::new(r"^(?:会话日期|conversation date|session date)[：:]\s*(\d{4}年\d{1,2}月)")
            .expect("Chinese session month-year regex")
    });
    value.lines().map(str::trim).find_map(|line| {
        if let Some(captures) = english.captures(line) {
            return captures
                .get(1)
                .map(|capture| capture.as_str().trim().to_string());
        }
        let capture = chinese.captures(line)?.get(1)?;
        let next = line[capture.end()..].trim_start().chars().next();
        (!next.is_some_and(|character| character.is_ascii_digit() || character == '日'))
            .then(|| capture.as_str().trim().to_string())
    })
}

fn temporal_granularity_accepts_month_year(granularity: TemporalGranularity) -> bool {
    matches!(
        granularity,
        TemporalGranularity::Unspecified | TemporalGranularity::Year | TemporalGranularity::Month
    )
}

fn atom_satisfies_temporal_constraint(
    constraint: &TemporalConstraint,
    atom: &EvidenceAtom,
    temporal_context: &str,
    session_dates: &HashMap<u64, chrono::NaiveDate>,
) -> bool {
    if !constraint.is_active() {
        return true;
    }
    if atom.kind.grounding_role() != GroundingRole::Original {
        return false;
    }
    atom_temporal_dates(atom, temporal_context, session_dates)
        .into_iter()
        .any(|date| constraint.matches(date))
}

fn atom_temporal_dates(
    atom: &EvidenceAtom,
    temporal_context: &str,
    session_dates: &HashMap<u64, chrono::NaiveDate>,
) -> Vec<chrono::NaiveDate> {
    let temporal_context = if temporal_context.trim().is_empty() {
        atom.text.as_str()
    } else {
        temporal_context
    };
    let session_date = session_dates.get(&atom.memory_id).copied();
    let mut dates = parse_temporal_dates(&atom.text);
    if dates.is_empty() {
        if let Some(session_date) = session_date {
            dates.extend(exact_relative_event_dates(&atom.text, session_date));
        }
    }
    if dates.is_empty() && temporal_context != atom.text {
        dates.extend(parse_temporal_dates(temporal_context));
        if dates.is_empty() {
            if let Some(session_date) = session_date {
                dates.extend(exact_relative_event_dates(temporal_context, session_date));
            }
        }
    }
    if dates.is_empty() {
        if let Some(session_date) = session_date {
            if let Some((derived, _, _)) = crate::memory_engine_summary::relative_event_date_answer(
                temporal_context,
                session_date,
            ) {
                dates.extend(parse_temporal_dates(&derived));
            }
        }
    }
    dates.sort_unstable();
    dates.dedup();
    dates
}

// Evidence selection needs an exact date, while relative_event_date_answer intentionally
// returns readable prose such as "the Friday before 16 November 2023".
fn exact_relative_event_dates(
    value: &str,
    session_date: chrono::NaiveDate,
) -> Vec<chrono::NaiveDate> {
    let lower = value.to_lowercase();
    let session_weekday = i64::from(session_date.weekday().number_from_monday());

    for (marker, target_weekday) in [
        ("last sunday", 7_i64),
        ("last sun", 7),
        ("last monday", 1),
        ("last mon", 1),
        ("last tuesday", 2),
        ("last tue", 2),
        ("last wednesday", 3),
        ("last wed", 3),
        ("last thursday", 4),
        ("last thu", 4),
        ("last friday", 5),
        ("last fri", 5),
        ("last saturday", 6),
        ("last sat", 6),
    ] {
        if contains_bounded_ascii_phrase(&lower, marker) {
            let days_back = (session_weekday - target_weekday).rem_euclid(7);
            let days_back = if days_back == 0 { 7 } else { days_back };
            return session_date
                .checked_sub_signed(chrono::Duration::days(days_back))
                .into_iter()
                .collect();
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
        if contains_bounded_ascii_phrase(&lower, marker) {
            let days_back = (session_weekday - target_weekday).rem_euclid(7);
            return session_date
                .checked_sub_signed(chrono::Duration::days(days_back))
                .into_iter()
                .collect();
        }
    }

    if contains_any(&lower, &["last week", "previous week"]) {
        return session_date
            .checked_sub_signed(chrono::Duration::days(7))
            .into_iter()
            .collect();
    }
    if contains_any(&lower, &["next week", "following week"]) {
        return session_date
            .checked_add_signed(chrono::Duration::days(7))
            .into_iter()
            .collect();
    }

    Vec::new()
}

fn parse_temporal_dates(value: &str) -> Vec<chrono::NaiveDate> {
    date_regex()
        .find_iter(value)
        .filter_map(|candidate| {
            crate::memory_engine_summary::source_session_date(&format!(
                "Session date: {}",
                candidate.as_str()
            ))
        })
        .collect()
}

fn atom_has_event_time_support(
    plan: &QueryPlan,
    atom: &EvidenceAtom,
    temporal_context: &str,
    session_dates: &HashMap<u64, chrono::NaiveDate>,
    session_month_years: &HashMap<u64, String>,
) -> bool {
    let temporal_context = if temporal_context.trim().is_empty() {
        atom.text.as_str()
    } else {
        temporal_context
    };
    if contains_date_like_value(temporal_context)
        || contains_bare_calendar_month_reference(temporal_context)
    {
        return true;
    }
    if let Some(session_date) = session_dates.get(&atom.memory_id) {
        return crate::memory_engine_summary::relative_event_date_answer(
            temporal_context,
            *session_date,
        )
        .is_some();
    }
    let _ = (plan, session_month_years);
    false
}

fn has_unresolved_temporal_reference(value: &str) -> bool {
    let lower = value.to_lowercase();
    contains_any(
        &lower,
        &[
            "last season",
            "previous season",
            "earlier season",
            "sometime",
            "some time",
            "recently",
            "earlier",
            "long ago",
            "during ",
            "while ",
        ],
    )
}

fn contains_any(value: &str, candidates: &[&str]) -> bool {
    candidates.iter().any(|candidate| value.contains(candidate))
}

fn contains_bounded_ascii_phrase(value: &str, phrase: &str) -> bool {
    value.match_indices(phrase).any(|(index, _)| {
        let before = value[..index].chars().next_back();
        let after = value[index + phrase.len()..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric())
            && after.is_none_or(|character| !character.is_ascii_alphanumeric())
    })
}

fn contains_whole_ascii_word(value: &str, word: &str) -> bool {
    value.match_indices(word).any(|(index, _)| {
        let before = value[..index].chars().next_back();
        let after = value[index + word.len()..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric())
            && after.is_some_and(|character| character.is_ascii_whitespace())
    })
}

fn is_cjk(character: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&character)
}

fn is_question_word(value: &str) -> bool {
    matches!(
        value,
        "a" | "an"
            | "all"
            | "analyze"
            | "and"
            | "are"
            | "about"
            | "can"
            | "cite"
            | "could"
            | "continue"
            | "did"
            | "do"
            | "does"
            | "explain"
            | "provide"
            | "tell"
            | "show"
            | "summarize"
            | "describe"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "how"
            | "in"
            | "infer"
            | "is"
            | "it"
            | "list"
            | "may"
            | "might"
            | "not"
            | "of"
            | "on"
            | "or"
            | "please"
            | "should"
            | "the"
            | "this"
            | "their"
            | "they"
            | "to"
            | "was"
            | "were"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "with"
            | "will"
            | "would"
            | "什么"
            | "多少"
            | "哪些"
            | "怎么"
            | "如何"
            | "是否"
            | "关于"
            | "一下"
            | "讲了"
            | "发生"
            | "事情"
            | "内容"
            | "最近"
            | "以及"
            | "answer"
            | "direct"
            | "evidence"
            | "available"
            | "record"
            | "records"
            | "likely"
            | "yes"
            | "no"
            | "based"
    )
}

fn word_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"[\p{L}\p{N}'_-]+").expect("answer query token regex"))
}

fn subject_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)\b(?:did|does|is|was|were|has|have|about|for|with)\s+([a-z][a-z'-]{1,})")
            .expect("answer query subject regex")
    })
}
fn person_name_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\b[A-Z][a-z]+(?:[-'][A-Z]?[a-z]+)?(?:['’]s)?\b")
            .expect("answer query person name regex")
    })
}

fn named_entity_sequence_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\b[A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*(?:['’]s)?\b")
            .expect("answer query named-entity sequence regex")
    })
}

fn open_plural_list_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        // "What books has Tim read?", "What are Joanna's hobbies?" (plural
        // verb before a possessed plural), "What people has Maria met?"
        // (irregular plural).
        Regex::new(r"(?i)\b(?:what|which)(?:\s+[a-z0-9][a-z0-9'-]*){0,4}\s+[a-z][a-z'-]*s\s+(?:did|does|do|has|have|are|were|was|is|can|could|would|will)\b|\b(?:what|which)\s+(?:are|were)\s+[a-z]+(?:'s|’s|s')\s+(?:[a-z-]+\s+)?[a-z][a-z-]*s\b|\b(?:what|which)\s+(?:people|children|men|women)\b|\b(?:what|which)\s+(?:some\s+)?(?:kinds|types|sorts)\s+of\b|\bwhat\s+(?:are|were)\s+some\b|\b(?:what|which)\s+(?:[a-z-]+\s+){0,2}(?:things|ways|places|activities|events|traits|changes|items|foods|games|hobbies|symbols|books|movies|songs|artists|bands|pets|plans|goals|gifts|recipes|instruments|sports|countries|cities)\b")
            .expect("answer query open plural list regex")
    })
}

fn speaker_line_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^(?:D\d+:\d+\s+)?([^:\n]{1,64}):\s*(.+)$")
            .expect("answer evidence speaker regex")
    })
}

fn dialogue_turn_prefix_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)^(D\d+:\d+)\s+").expect("answer evidence dialogue turn anchor regex")
    })
}

fn media_caption_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"(?i)\[(image|video|audio)(?:\s+(?:caption|description))?\s*:\s*([^\]]+)\]")
            .expect("answer evidence media regex")
    })
}

fn date_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"(?i)(?:\b\d{4}-\d{1,2}-\d{1,2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)(?:,?\s+\d{4})?\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{4})?\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}\b|\b(?:19|20)\d{2}\b|\b\d{4}年\d{1,2}月(?:\d{1,2}日)?)",
        )
        .expect("answer evidence date regex")
    })
}

fn location_value_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"\b(?i:in|at|to|from|near|around|across|through|within|outside|inside|visiting|visited|traveling|travelled)\s+(?:the\s+)?([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*)",
        )
        .expect("answer evidence location regex")
    })
}

fn named_location_value_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"\b(?i:(?:home\s+)?(?:city|country|state|province|region|town|village|place))\s*(?:is|was|,|:)\s*(?:the\s+)?([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*)",
        )
        .expect("named location value regex")
    })
}

fn coordinated_proper_value_pair_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"\b([A-Z][A-Za-z'-]*(?:\s+(?:[A-Z][A-Za-z'-]*|of|the|a|an|to|for|in|on|at))*?)\s+(?:and|or)\s+([A-Z][A-Za-z'-]*(?:\s+(?:[A-Z][A-Za-z'-]*|of|the|a|an|to|for|in|on|at))*)\s+(?:are|were|include|includes|included|remain|remains|seem|seems)\b",
        )
        .expect("coordinated proper-value pair regex")
    })
}

fn answer_first_location_value_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(
            r"\b(?i:it(?:'s|\s+is)|that(?:'s|\s+is)|this(?:'s|\s+is)|the\s+(?:city|location|place)\s+is)\s+([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*)*)\s*(?:,|[.!;]|$)",
        )
        .expect("answer-first location regex")
    })
}

fn citation_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"\[[^\]\r\n]*memory#[0-9]+[^\]\r\n]*\]").expect("answer claim citation regex")
    })
}

fn number_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| Regex::new(r"\b[0-9]+(?:\.[0-9]+)?\b").expect("answer number regex"))
}


fn source_condition_clause(source_lower: &str) -> Option<(ConditionRelation, &str)> {
    source_condition_parts(source_lower).map(|(_, _, relation, tail)| (relation, tail))
}

fn source_condition_parts(source_lower: &str) -> Option<(&str, &str, ConditionRelation, &str)> {
    let markers = [
        ("only after", ConditionRelation::OnlyAfter),
        ("only before", ConditionRelation::OnlyBefore),
        ("as long as", ConditionRelation::If),
        ("only when", ConditionRelation::OnlyWhen),
        ("only if", ConditionRelation::OnlyWhen),
        ("provided that", ConditionRelation::Assuming),
        ("assuming", ConditionRelation::Assuming),
        ("unless", ConditionRelation::Unless),
        ("because", ConditionRelation::Causal),
        ("due to", ConditionRelation::Causal),
        ("since", ConditionRelation::Causal),
        ("if", ConditionRelation::If),
        ("when", ConditionRelation::When),
        ("仅当", ConditionRelation::OnlyWhen),
        ("除非", ConditionRelation::Unless),
        ("因为", ConditionRelation::Causal),
        ("由于", ConditionRelation::Causal),
        ("如果", ConditionRelation::If),
        ("当", ConditionRelation::If),
    ];
    let mut first = None;
    for (marker, relation) in markers {
        for (start, _) in source_lower.match_indices(marker) {
            if condition_marker_is_embedded_lexeme(source_lower, start, marker) {
                continue;
            }
            if !condition_marker_has_boundaries(source_lower, start, marker.len()) {
                continue;
            }
            if first.is_none_or(|(previous, _, _)| start < previous) {
                first = Some((start, marker.len(), relation));
            }
        }
    }
    first.map(|(start, marker_len, relation)| {
        (
            &source_lower[..start],
            &source_lower[start..start + marker_len],
            relation,
            &source_lower[start + marker_len..],
        )
    })
}

fn condition_relation_is_compatible(
    requested: ConditionRelation,
    observed: ConditionRelation,
) -> bool {
    match requested {
        ConditionRelation::None => true,
        ConditionRelation::When => observed == ConditionRelation::When,
        ConditionRelation::OnlyAfter => observed == ConditionRelation::OnlyAfter,
        ConditionRelation::OnlyBefore => observed == ConditionRelation::OnlyBefore,
        ConditionRelation::If => {
            matches!(
                observed,
                ConditionRelation::If | ConditionRelation::Assuming
            )
        }
        ConditionRelation::Unless => observed == ConditionRelation::Unless,
        ConditionRelation::OnlyWhen => observed == ConditionRelation::OnlyWhen,
        ConditionRelation::Assuming => {
            matches!(
                observed,
                ConditionRelation::Assuming | ConditionRelation::If
            )
        }
        ConditionRelation::Causal => observed == ConditionRelation::Causal,
    }
}

fn strict_condition_terms_match_evidence(plan: &QueryPlan, source_lower: &str) -> bool {
    if plan.condition_relation == ConditionRelation::None {
        return plan.condition_terms.is_empty();
    }
    let Some((observed_relation, condition_clause)) = source_condition_clause(source_lower) else {
        return false;
    };
    condition_relation_is_compatible(plan.condition_relation, observed_relation)
        && plan
            .condition_terms
            .iter()
            .all(|term| text_matches_lexical_term(condition_clause, term))
}

fn negation_scope_clauses(value: &str) -> Vec<&str> {
    let mut clauses = value.split(['.', ';', '!', '?', '\n']).collect::<Vec<_>>();
    for separator in [
        " and ",
        " but ",
        " however ",
        " instead ",
        " although ",
        " after ",
        " before ",
        " while ",
        " because ",
        " due to ",
        " unless ",
        " if ",
        " when ",
        " then ",
        "但是",
        "不过",
        "而是",
        "之后",
        "之前",
        "因为",
        "由于",
        "如果",
        "除非",
        "当",
    ] {
        clauses = clauses
            .into_iter()
            .flat_map(|clause| clause.split(separator))
            .collect();
    }
    clauses
}

fn negation_token(word: &str) -> bool {
    matches!(
        word,
        "not"
            | "never"
            | "no"
            | "without"
            | "except"
            | "cannot"
            | "can't"
            | "didn't"
            | "doesn't"
            | "wasn't"
            | "weren't"
    )
}

fn negated_predicate_skip_word(word: &str) -> bool {
    matches!(
        word,
        "a" | "an"
            | "the"
            | "to"
            | "be"
            | "been"
            | "being"
            | "am"
            | "are"
            | "is"
            | "was"
            | "were"
            | "do"
            | "does"
            | "did"
            | "have"
            | "has"
            | "had"
            | "can"
            | "could"
            | "will"
            | "would"
            | "may"
            | "might"
            | "must"
            | "should"
            | "really"
            | "actually"
            | "ever"
            | "yet"
            | "even"
            | "just"
            | "only"
    )
}

fn negation_scope_matches_plan(
    plan: &QueryPlan,
    atom_lower: &str,
    claim_content_terms: &[String],
) -> bool {
    let value_terms = claim_content_terms
        .iter()
        .filter(|term| {
            !plan.condition_terms.iter().any(|condition| {
                text_matches_lexical_term(term, condition)
                    || text_matches_lexical_term(condition, term)
            }) && !matches!(
                term.as_str(),
                "if" | "when"
                    | "unless"
                    | "only"
                    | "provided"
                    | "assuming"
                    | "because"
                    | "due"
                    | "since"
                    | "如果"
                    | "当"
                    | "除非"
                    | "仅当"
            )
        })
        .collect::<Vec<_>>();
    let value_terms = if value_terms.is_empty() {
        claim_content_terms.iter().collect::<Vec<_>>()
    } else {
        value_terms
    };

    negation_scope_clauses(atom_lower)
        .into_iter()
        .any(|clause| {
            let chinese_negated = ["不是", "没有", "从未", "不能", "不包括", "除外"]
                .iter()
                .any(|marker| clause.contains(marker));
            let words = word_regex()
                .find_iter(clause)
                .map(|value| value.as_str().to_ascii_lowercase())
                .collect::<Vec<_>>();
            let negation_indexes = words
                .iter()
                .enumerate()
                .filter_map(|(index, word)| negation_token(word).then_some(index))
                .collect::<Vec<_>>();
            if !chinese_negated && negation_indexes.is_empty() {
                return false;
            }
            if words.windows(2).any(|pair| {
                matches!(pair[0].as_str(), "not" | "never")
                    && matches!(pair[1].as_str(), "just" | "only")
            }) {
                return false;
            }
            let predicate_matches = if chinese_negated {
                plan.action_terms.is_empty()
                    || plan
                        .action_terms
                        .iter()
                        .any(|term| text_matches_lexical_term(clause, term))
            } else {
                let Some(negation_index) = negation_indexes.into_iter().next() else {
                    return false;
                };
                let Some(predicate) = words
                    .iter()
                    .skip(negation_index + 1)
                    .find(|word| !negated_predicate_skip_word(word))
                else {
                    return false;
                };
                plan.action_terms.is_empty()
                    || plan
                        .action_terms
                        .iter()
                        .any(|term| text_matches_term(predicate, term))
            };
            predicate_matches
                && value_terms
                    .iter()
                    .all(|term| text_matches_lexical_term(clause, term))
        })
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryLayer;
    use chrono::Utc;

    fn test_memory(id: u64, content: &str) -> MemoryRecord {
        MemoryRecord {
            id,
            conversation_id: "operator-plan-test".to_string(),
            layer: MemoryLayer::Fact,
            content: content.to_string(),
            tokens: Vec::new(),
            source: "unit-test".to_string(),
            trust: 1.0,
            created_at: Utc::now(),
            anchor_ids: Vec::new(),
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
            version: 3,
            event_metadata: None,
            procedure_schema: None,
            procedure_execution_state: None,
            procedure_governance: None,
            retrieval_hits: 0,
            last_retrieved_at: None,
            embedding_provider: None,
            embedding_model: None,
            embedding_status: None,
            embedded_at: None,
            semantic_node: None,
            semantic_conflict: None,
            episodic_metadata: None,
            meta_memory_state: None,
            activation_metadata: None,
        }
    }

    fn test_item(id: u64, score: f32, content: &str) -> RetrievedItem {
        RetrievedItem {
            id,
            layer: MemoryLayer::Fact,
            memory_type: "source_record".to_string(),
            content: content.to_string(),
            score,
            source: format!("source-{id}"),
            entity: None,
            created_at: Utc::now(),
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
            semantic_node: None,
            semantic_conflict: None,
            canonical_semantic_id: None,
        }
    }


    #[test]
    fn model_pool_preserves_full_atoms_and_rejects_mixed_source_versions() {
        let body = "complete original evidence ".repeat(200);
        let original = test_item(1, 0.9, &format!("John: {body}"));
        let atoms = reasoning_source_atoms(&[original.clone(), original.clone()]).unwrap();
        let statement = atoms
            .values()
            .find(|atom| atom.evidence_kind == "body_fact")
            .unwrap();
        assert_eq!(statement.body, body.trim());
        assert_eq!(statement.speaker.as_deref(), Some("John"));
        assert_eq!(statement.source_version, original.version);
        assert!(!statement.descriptive_only);
        let mut changed = original.clone();
        changed.version += 1;
        assert_eq!(
            reasoning_source_atoms(&[original.clone(), changed]).unwrap_err(),
            "source_snapshot_changed"
        );
        let mut changed = original.clone();
        changed.content.push_str(" changed without version bump");
        assert_eq!(
            reasoning_source_atoms(&[original, changed]).unwrap_err(),
            "source_snapshot_changed"
        );
    }

    #[test]
    fn possessive_relation_actor_differs_from_coordinated_participant() {
        assert_eq!(
            possessive_relation_actors("when did jolene's parents give her first console?"),
            vec!["parents".to_string()]
        );
        assert!(possessive_relation_actors("when did jolene and her partner try scuba diving?")
            .is_empty());
    }

    #[test]
    fn superlative_number_of_is_a_value_question_not_a_count() {
        assert!(!count_question_requested(
            "what was the highest number of points john scored in a game recently?"
        ));
        assert!(count_question_requested("how many times has john injured his ankle?"));
        assert!(count_question_requested("what is the number of games john played?"));
    }

    #[test]
    fn content_free_reactions_are_low_information() {
        for filler in ["Wow, Calvin!", "Nice job, Jolene!", "Wow that's so cool!", "Awesome, thanks Dave!"] {
            assert!(is_low_information_statement(filler), "{filler}");
        }
        for content in [
            "Wow, that's a great necklace!",
            "Yeah, birds are amazing!",
            "I scored 40 points last week.",
            "Definitely Star Wars!",
            "Definitely Batman!",
        ] {
            assert!(!is_low_information_statement(content), "{content}");
        }
    }

    #[test]
    fn leading_direct_address_does_not_steal_the_speakers_facts() {
        for (speaker, statement) in [
            ("Caroline", "Yep, Melanie! I've got some other stuff with sentimental value, like my hand-painted bowl. A friend made it for my 18th birthday ten years ago."),
            ("Melanie", "Caroline, so glad you got the support! This book I read last year reminds me to always pursue my dreams."),
            ("Melanie", "Hey Caroline, hope all's good! I had a quiet weekend after we went camping with my fam two weekends ago."),
            ("Caroline", "I'm always here for you, Mel! We had a blast last year at the Pride fest."),
            ("Melanie", "Hey Caroline, that roadtrip this past weekend was insane! We were all freaked when my son got into an accident."),
        ] {
            let attribution = fact_attribution(statement, Some(speaker));
            assert!(
                attribution.fact_subjects.iter().any(|name| name == speaker),
                "{speaker}: {statement} -> {:?} / {:?}",
                attribution.fact_subjects,
                attribution.subject_binding
            );
        }
        // A capitalized sentence starter is not a person.
        let starter = fact_attribution(
            "Thanks, Melanie. It's definitely changed them. Some close friends kept supporting me.",
            Some("Caroline"),
        );
        assert!(!starter.fact_subjects.iter().any(|name| name == "Some"));
        assert!(!starter.mentioned_people.iter().any(|name| name == "Some"));
    }

    #[test]
    fn sentence_initial_gerund_phrase_is_not_a_person_name() {
        let gerund = "Organizing a basketball camp for kids last summer was awesome!";
        let attribution = fact_attribution(gerund, Some("John"));
        assert!(!attribution.fact_subjects.iter().any(|name| name == "Organizing"));
        assert!(attribution.fact_subjects.iter().any(|name| name == "John"));
        assert!(fact_attribution(
            "Organizing a basketball camp for kids in my hometown last summer was awesome!",
            Some("John")
        )
        .fact_subjects
        .iter()
        .any(|name| name == "John"));
        let name = "Ming visited the harbor last week.";
        assert!(fact_attribution(name, Some("John"))
            .fact_subjects
            .iter()
            .any(|person| person == "Ming"));
    }

    #[test]
    fn operator_plan_covers_all_domain_renderers_and_terminal_gate() {
        let plan = build_query_plan("What happened?", &[]);
        let operators = build_operator_plan(&plan, true);
        let names = operators
            .iter()
            .map(|operator| operator.as_str())
            .collect::<HashSet<_>>();
        assert_eq!(operators.len(), 29);
        assert_eq!(names.len(), operators.len());
        assert_eq!(operators.last(), Some(&AnswerOperator::AnswerabilityGate));
        assert!(names.contains("operational_habit_v1"));
        assert!(names.contains("learning_resume_v1"));
        assert!(names.contains("document_comparison_v1"));
        assert!(names.contains("experience_recall_v1"));
        assert!(names.contains("design_style_v1"));
        assert!(names.contains("multimodal_appearance_change_v1"));
        assert!(names.contains("temporal_reasoning_v1"));
        assert!(names.contains("novel_style_v1"));
        assert!(names.contains("unified_answer_generator_v1"));
    }

    #[test]
    fn explicit_selection_plan_excludes_full_space_record_operators() {
        let plan = build_query_plan("What happened?", &[]);
        let operators = build_operator_plan(&plan, false);
        assert!(!operators.contains(&AnswerOperator::NovelChapterRangeFromRecords));
        assert!(!operators.contains(&AnswerOperator::NovelChapterFromRecords));
        assert_eq!(operators.last(), Some(&AnswerOperator::AnswerabilityGate));
    }

    #[test]
    fn persisted_projection_binds_memory_scope_and_source_version() {
        let atoms = persisted_evidence_atoms_for_memory(&test_memory(
            73,
            "Session date: 2026-08-23\nMelanie: I painted a blue wall.",
        ));
        assert!(!atoms.is_empty());
        assert!(atoms.iter().all(|atom| atom.memory_id == 73));
        assert!(atoms
            .iter()
            .all(|atom| atom.conversation_id == "operator-plan-test"));
        assert!(atoms.iter().all(|atom| atom.source_version == 3));
        assert!(atoms
            .iter()
            .all(|atom| atom.atom_id.starts_with("memory:73:atom:")));
        let dialogue = atoms
            .iter()
            .find(|atom| atom.speaker.as_deref() == Some("Melanie"))
            .expect("dialogue atom");
        assert_eq!(
            dialogue.metadata["projection_format"],
            "typed_evidence_atom_v3_reasoning_source"
        );
        assert_eq!(
            dialogue.metadata["attribution"]["fact_subjects"],
            serde_json::json!(["Melanie"])
        );
        assert!(dialogue.metadata["attribution"]["subject_binding"]
            .as_array()
            .is_some_and(|bindings| !bindings.is_empty()));
    }

    #[test]
    fn persisted_event_scaffold_never_invents_quantity_or_identity() {
        let mut memory = test_memory(73, "event: John inspected twice.");
        let project = |memory: &MemoryRecord| {
            persisted_evidence_atoms_for_memory(memory).into_iter()
                .find(|atom| atom.atom_kind == "event").unwrap()
        };
        let original = project(&memory);
        let scaffold = &original.metadata["reasoning_source_scaffold"];
        assert_eq!(scaffold["semantic_status"], "source_scaffold_unverified");
        assert_eq!(scaffold["relations"], serde_json::json!([]));
        let observation = &scaffold["event_observations"][0];
        assert!(observation["multiplicity"].is_null());
        assert!(observation["native_event_identity"].is_null());
        assert_eq!(observation["source_span"]["text"], original.body);
        assert_eq!(observation["source_span"]["end"], original.body.len());
        assert_eq!(observation["merge_basis"]["cross_observation_identity"], "unresolved");
        let id = observation["observation_id"].clone();
        assert_eq!(project(&memory).metadata["reasoning_source_scaffold"]["event_observations"][0]["observation_id"], id);
        memory.version += 1;
        assert_ne!(project(&memory).metadata["reasoning_source_scaffold"]["event_observations"][0]["observation_id"], id);
        memory.version -= 1;
        memory.content.push_str(" A correction.");
        assert_ne!(project(&memory).metadata["reasoning_source_scaffold"]["event_observations"][0]["observation_id"], id);
    }

    #[test]
    fn query_plan_exposes_subject_time_list_and_inference_slots() {
        let plan = build_query_plan(
            "Which projects did Melanie most recently make, and what was likely shared?",
            &[],
        );
        assert_eq!(plan.subject.as_deref(), Some("Melanie"));
        assert_eq!(plan.temporal_scope, "latest");
        assert_eq!(plan.cardinality, Cardinality::List);
        assert_eq!(plan.inference_type, InferenceType::Likely);
        assert!(plan.required_slots.contains(&"subject".to_string()));
        assert!(plan.required_slots.contains(&"inference_basis".to_string()));
    }

    #[test]
    fn count_and_list_intents_use_grammatical_boundaries() {
        let country = build_query_plan("Which country was Tim visiting?", &[]);
        assert_eq!(country.cardinality, Cardinality::Single);
        assert_eq!(country.answer_target, AnswerTarget::Location);

        let challenge = build_query_plan("What challenge did John encounter?", &[]);
        assert_eq!(challenge.cardinality, Cardinality::Single);
        assert_ne!(challenge.answer_target, AnswerTarget::Count);

        let novels = build_query_plan("Which two fantasy novels does Tim enjoy?", &[]);
        assert_eq!(novels.cardinality, Cardinality::List);

        let basketball =
            build_query_plan("What did John give to the basketball players he met?", &[]);
        assert_eq!(basketball.cardinality, Cardinality::Single);
        assert_eq!(basketball.answer_target, AnswerTarget::Object);
    }

    #[test]
    fn question_plan_separates_grammar_objects_actions_and_temporal_qualifiers() {
        let married = build_query_plan("When did John get married at a greenhouse?", &[]);
        assert_eq!(married.action_terms, vec!["married"]);
        assert_eq!(married.object_terms, vec!["greenhouse"]);

        let injury = build_query_plan("When did John get an ankle injury in 2023?", &[]);
        assert!(injury.action_terms.is_empty());
        assert_eq!(injury.object_terms, vec!["ankle", "injury"]);
        assert!(injury.quantity_terms.is_empty());
        assert!(injury.time_terms.contains(&"2023".to_string()));

        let reading = build_query_plan(
            "Which book was John reading during his recovery from an ankle injury?",
            &[],
        );
        assert_eq!(reading.action_terms, vec!["reading"]);
        assert!(reading.object_terms.is_empty());
        assert_eq!(reading.qualifier_terms, vec!["ankle", "injury", "recovery"]);
        assert!(reading.subject_is_action_agent);
        assert!(reading
            .required_slots
            .contains(&"qualifier_context".to_string()));

        let surfing = build_query_plan("How does John feel while surfing?", &[]);
        assert_eq!(surfing.answer_target, AnswerTarget::Manner);
        assert_eq!(surfing.action_terms, vec!["feel"]);
        assert_eq!(surfing.qualifier_terms, vec!["surfing"]);
        assert!(surfing.object_terms.is_empty());

        let gift = build_query_plan(
            "What did John's teammates give him when they met on Aug 15th?",
            &[],
        );
        assert_eq!(gift.answer_target, AnswerTarget::Object);
        assert!(gift.time_terms.contains(&"15th".to_string()));
        assert!(gift.quantity_terms.is_empty());
        assert_eq!(gift.action_terms, vec!["give"]);
        assert_eq!(gift.qualifier_terms, vec!["met"]);
        assert_eq!(gift.object_terms, vec!["teammates"]);
        assert!(!gift.subject_is_action_agent);

        let temporal = build_query_plan("When did John achieve a career-high?", &[]);
        assert_eq!(temporal.answer_target, AnswerTarget::Temporal);
        assert_eq!(required_object_overlap(4), 1);

        let capsule = build_query_plan(
            "Continue the imported Memory Capsule project for capsule-portability-token-6174 from the resume context and show trace evidence.",
            &[],
        );
        assert!(capsule.subject.is_none(), "{capsule:?}");
        assert!(capsule.named_entities.is_empty(), "{capsule:?}");
    }
    #[test]
    fn question_plan_does_not_require_open_event_prompt_fragments_as_objects() {
        let chapter = build_query_plan("第315章发生了什么？", &[]);
        assert!(chapter.query_terms.is_empty(), "{:?}", chapter.query_terms);
        assert_eq!(chapter.structural_selectors, vec!["chapter:315"]);
        assert!(chapter
            .required_slots
            .contains(&"structural_selector".to_string()));
        assert!(
            chapter.object_terms.is_empty(),
            "{:?}",
            chapter.object_terms
        );
        assert!(!chapter.required_slots.contains(&"object".to_string()));

        let incident = build_query_plan("杭州仓库发生了什么？", &[]);
        assert!(incident.query_terms.contains(&"杭州".to_string()));
        assert!(incident.query_terms.contains(&"仓库".to_string()));
        assert!(incident.object_terms.contains(&"杭州".to_string()));
        assert!(incident.object_terms.contains(&"仓库".to_string()));
        assert!(!incident
            .object_terms
            .iter()
            .any(|term| matches!(term.as_str(), "库发" | "发生" | "生了" | "了什" | "什么")));

        let deployment = build_query_plan("What happened during the database migration?", &[]);
        assert!(deployment.query_terms.contains(&"database".to_string()));
        assert!(deployment.query_terms.contains(&"migration".to_string()));
        assert!(!deployment.query_terms.contains(&"happened".to_string()));
        let section = build_query_plan("What happened in section 7?", &[]);
        assert_eq!(section.structural_selectors, vec!["section:7"]);
        assert!(section
            .required_slots
            .contains(&"structural_selector".to_string()));
    }
    #[test]
    fn structural_selector_prefers_matching_general_document_scope() {
        let pipeline = plan_answer_pipeline(
            "What happened in section 7?",
            &[
                test_item(6, 0.95, "Section 6: The signed release was deployed."),
                test_item(
                    7,
                    0.55,
                    "Section 7: The certificate expired and the deployment was rolled back.",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(pipeline.receipt.evidence_plan.statement_candidates, 2);
        assert_eq!(
            pipeline.receipt.evidence_plan.original_evidence_candidates,
            2
        );
        assert_eq!(pipeline.receipt.evidence_plan.selected_atom_count, 1);
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[7],
            "{:#?}",
            pipeline.receipt
        );
        let answer = pipeline.unified_draft.expect("section-scoped answer");
        assert!(
            answer.text.contains("certificate expired"),
            "{}",
            answer.text
        );
        assert!(pipeline
            .receipt
            .slot_coverage
            .iter()
            .any(|slot| { slot.slot == "structural_selector" && slot.status == "covered" }));
    }

    #[test]
    fn multiple_structural_scopes_require_complementary_general_document_evidence() {
        let pipeline = plan_answer_pipeline(
            "What happened in sections 7 and 8?",
            &[
                test_item(6, 0.98, "Section 6: The signed release was deployed."),
                test_item(7, 0.56, "Section 7: The certificate expired."),
                test_item(8, 0.54, "Section 8: The certificate was renewed."),
            ],
            3,
            3,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(
            pipeline.receipt.query_plan.structural_selectors,
            vec!["section:7", "section:8"]
        );
        assert_eq!(pipeline.receipt.query_plan.answer_target, "event");
        assert_eq!(pipeline.receipt.query_plan.answer_shape, "action_set");
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[7, 8],
            "{:#?}",
            pipeline.receipt
        );
        for selector in ["section:7", "section:8"] {
            assert!(pipeline.receipt.slot_coverage.iter().any(|slot| {
                slot.slot == structural_selector_slot_name(selector) && slot.status == "covered"
            }));
        }
    }

    #[test]
    fn compressed_cjk_structural_scopes_select_complementary_document_evidence() {
        let pipeline = plan_answer_pipeline(
            "第7、8章发生了什么？",
            &[
                test_item(6, 0.98, "第6章：签名版本已部署。"),
                test_item(7, 0.56, "第7章：证书已经过期。"),
                test_item(8, 0.54, "第8章：证书已经更新。"),
            ],
            3,
            3,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(
            pipeline.receipt.query_plan.structural_selectors,
            vec!["chapter:7", "chapter:8"]
        );
        assert_eq!(pipeline.receipt.query_plan.answer_target, "event");
        assert_eq!(pipeline.receipt.query_plan.answer_shape, "action_set");
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[7, 8],
            "{:#?}",
            pipeline.receipt
        );
        for selector in ["chapter:7", "chapter:8"] {
            assert!(pipeline.receipt.slot_coverage.iter().any(|slot| {
                slot.slot == structural_selector_slot_name(selector) && slot.status == "covered"
            }));
        }
    }

    #[test]
    fn projection_separates_date_dialogue_and_native_media() {
        let atoms = project_source(ProjectionInput {
            memory_id: 41,
            content: "Session date: 13 October, 2023\nD17:12 Melanie: I painted a sunset. [Image caption: a pink sunset over water]",
            source: "conversation-17",
            entity: Some("Melanie"),
            score: 0.8,
            is_event: false,
        });
        assert!(atoms
            .iter()
            .any(|atom| atom.kind == EvidenceKind::SessionDate));
        assert!(atoms.iter().any(|atom| atom.kind == EvidenceKind::Statement
            && atom.speaker.as_deref() == Some("Melanie")));
        assert!(atoms.iter().any(|atom| {
            atom.kind == EvidenceKind::Statement
                && atom.source_anchor.as_ref().is_some_and(|anchor| {
                    anchor.kind == "dialogue_turn" && anchor.value == "D17:12"
                })
        }));
        assert!(atoms
            .iter()
            .any(|atom| atom.kind == EvidenceKind::NativeMedia && atom.modality == "image"));
        assert!(atoms.iter().any(|atom| {
            atom.kind == EvidenceKind::NativeMedia
                && atom
                    .source_anchor
                    .as_ref()
                    .is_some_and(|anchor| anchor.value == "D17:12")
        }));
        assert!(atoms.iter().any(|atom| atom.kind == EvidenceKind::Entity));
    }

    #[test]
    fn structured_profile_scope_and_described_modality_are_typed_source_atoms() {
        let content = serde_json::json!({
            "memory_type": "operational_habit_profile",
            "mission_id": "rover-habit-alpha",
            "operator": "mission-control",
            "dimension": "telemetry_attention",
            "summary": "Telemetry attention favors source-visible threshold readings.",
            "modality_signal_records": [{
                "modality": "sensor",
                "source": "battery-dashboard",
                "observation": "Battery and lidar warnings were visible before action.",
                "signal_kinds": ["sensor_or_telemetry", "structured_metadata"]
            }]
        })
        .to_string();
        let atoms = project_source(ProjectionInput {
            memory_id: 301,
            content: &content,
            source: "operational-profile-test",
            entity: None,
            score: 0.9,
            is_event: false,
        });
        let original = atoms
            .iter()
            .filter(|atom| atom.kind.grounding_role() == GroundingRole::Original)
            .collect::<Vec<_>>();
        assert!(!original.is_empty());
        assert!(original
            .iter()
            .all(|atom| atom_has_verified_fact_subject(atom, "rover-habit-alpha")));
        assert!(original.iter().all(|atom| atom.subject_binding.iter().any(|binding| {
            binding.starts_with("structured_profile_scope:mission_id:")
        })));
        let sensor = original
            .iter()
            .find(|atom| atom.modality == "sensor")
            .expect("described sensor observation atom");
        assert_eq!(sensor.kind, EvidenceKind::Statement);
        assert!(sensor.text.contains("Described sensor evidence"));
        assert!(!original
            .iter()
            .any(|atom| atom.kind == EvidenceKind::NativeMedia));

        let unrelated = serde_json::json!({
            "memory_type": "source_record",
            "source_id": "rover-habit-alpha",
            "summary": "An unrelated source record."
        })
        .to_string();
        let unrelated_atoms = project_source(ProjectionInput {
            memory_id: 302,
            content: &unrelated,
            source: "ordinary-json-test",
            entity: None,
            score: 0.9,
            is_event: false,
        });
        assert!(unrelated_atoms
            .iter()
            .all(|atom| !atom_has_verified_fact_subject(atom, "rover-habit-alpha")));
    }

    #[test]
    fn profile_question_plan_keeps_scope_and_request_roles_distinct() {
        assert!(named_entity_is_request_actor(
            "What web design style should future UI work follow?",
            "UI"
        ));
        let writing = plan_answer_pipeline(
            "Analyze the writing style profile for launch-essay-style. Cite voice, diction, rhythm, tone, structure, pacing, and style transfer evidence.",
            &[test_item(
                311,
                0.9,
                &serde_json::json!({
                    "memory_type": "writing_style_profile",
                    "source_id": "launch-essay-style",
                    "dimension": "voice",
                    "summary": "The style is concise and evidence-first."
                })
                .to_string(),
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(writing.receipt.query_plan.subject.as_deref(), Some("launch-essay-style"));
        assert!(writing.receipt.query_plan.named_entities.is_empty());
        assert_eq!(writing.receipt.query_plan.requested_modalities, vec!["text"]);
        assert_eq!(writing.selected_memory_ids(), &[311], "{:#?}", writing.receipt);

        let operational = plan_answer_pipeline(
            "What operational habits and approval style should Physical AI use for rover-habit-alpha? Cite telemetry attention.",
            &[test_item(
                312,
                0.9,
                &serde_json::json!({
                    "memory_type": "operational_habit_profile",
                    "mission_id": "rover-habit-alpha",
                    "dimension": "telemetry_attention",
                    "summary": "Telemetry attention favors concrete thresholds.",
                    "modality_signal_records": [{
                        "modality": "sensor",
                        "source": "telemetry",
                        "observation": "The threshold reading was reviewed.",
                        "signal_kinds": ["sensor_or_telemetry"]
                    }]
                })
                .to_string(),
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(operational.receipt.query_plan.subject.as_deref(), Some("rover-habit-alpha"));
        assert!(operational.receipt.query_plan.named_entities.is_empty());
        assert!(operational
            .receipt
            .query_plan
            .requested_modalities
            .contains(&"sensor".to_string()));
        assert_eq!(operational.selected_memory_ids(), &[312], "{:#?}", operational.receipt);
        assert!(operational.receipt.slot_coverage.iter().any(|slot| {
            slot.slot == "requested_modality" && slot.status == "covered"
        }));

        let design = plan_answer_pipeline(
            "What web design style should future UI work follow?",
            &[test_item(
                313,
                0.9,
                &serde_json::json!({
                    "memory_type": "design_style_profile",
                    "source_id": "dashboard-design-style",
                    "dimension": "layout_density",
                    "summary": "Layout density favors compact, scannable operational surfaces over sparse marketing composition."
                })
                .to_string(),
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert!(design.receipt.query_plan.named_entities.is_empty());

        assert!(requested_modalities("play the voice recording")
            .contains(&"audio".to_string()));
    }

    #[test]
    fn answer_receipt_exposes_selected_original_source_anchor() {
        let pipeline = plan_answer_pipeline(
            "What did Melanie paint?",
            &[test_item(41, 0.8, "D17:12 Melanie: I painted a sunset.")],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(
            pipeline.receipt.evidence_plan.source_anchor_format,
            "answer_evidence_source_anchor_v1"
        );
        let selected_ids = pipeline
            .receipt
            .evidence_plan
            .selected_answer_atom_ids
            .iter()
            .collect::<HashSet<_>>();
        assert!(pipeline
            .receipt
            .evidence_plan
            .selected_atoms
            .iter()
            .any(|atom| {
                atom.grounding_role == "original"
                    && selected_ids.contains(&atom.atom_id)
                    && atom.source_anchor.as_ref().is_some_and(|anchor| {
                        anchor.kind == "dialogue_turn" && anchor.value == "D17:12"
                    })
            }));
    }
    #[test]
    fn same_turn_text_and_native_media_bind_slots_without_flattening_the_source() {
        let wedding_query = "When did John get married at a greenhouse?";
        let wedding = plan_answer_pipeline(
            wedding_query,
            &[test_item(
                12,
                0.8,
                "Session date: 2 October, 2023\nD12:2 John: I got married last week. [Image caption: a wedding ceremony in a greenhouse]",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(wedding.selected_memory_ids(), &[12]);
        let media = wedding
            .evidence_atoms
            .iter()
            .find(|atom| atom.kind == EvidenceKind::NativeMedia)
            .expect("wedding media atom");
        assert_eq!(media.speaker.as_deref(), Some("John"));
        assert!(media.supports_slots.contains(&"action".to_string()));
        assert!(media.supports_slots.contains(&"object".to_string()));
        assert!(media
            .supports_slots
            .contains(&"answer_bearing_fact".to_string()));

        let gift_query = "What did John's teammates give him when they met on Aug 15th?";
        let gift = plan_answer_pipeline(
            gift_query,
            &[test_item(
                7,
                0.8,
                "Session date: 15 August, 2023\nD7:7 John: Here is a photo of what my teammates gave me when we met. [Image caption: a basketball with autographs]",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        let draft = gift.unified_draft.expect("same-turn media answer");
        assert!(
            draft.text.contains("basketball with autographs"),
            "{}",
            draft.text
        );
        assert!(draft.text.contains("memory#7"), "{}", draft.text);
        assert!(!draft.text.contains("Here is a photo"), "{}", draft.text);
    }

    #[test]
    fn relational_qualifier_keeps_descriptive_answer_evidence() {
        let pipeline = plan_answer_pipeline(
            "How does John feel while surfing?",
            &[test_item(
                3,
                0.8,
                "D3:25 John: I had an awesome summer with my friends, surfing and riding the waves. The feeling was unreal!\nD3:26 Tim: Wow! How long have you been surfing?\nD3:27 John: I started surfing five years ago and it's been great.\nD3:28 Tim: The connection to nature must be incredible.\nD3:29 John: Being out in the water can be amazing. The waves, the wind, it's super exciting and free-feeling.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        let draft = pipeline.unified_draft.expect("surfing answer");
        assert!(draft.text.contains("free-feeling"), "{}", draft.text);
        assert!(draft.text.contains("memory#3"), "{}", draft.text);
    }

    #[test]
    fn relational_qualifier_bridges_distant_same_source_evidence_for_the_action_subject() {
        let pipeline = plan_answer_pipeline(
            "What was John reading during his recovery from an ankle injury?",
            &[test_item(
                19,
                0.8,
                "D19:1 Tim: I read a book about castles.\nD19:6 John: I hurt my ankle and stayed focused on my recovery.\nD19:7 Tim: I hope you feel better.\nD19:8 John: I am taking it one day at a time.\nD19:9 Tim: Basketball can be demanding.\nD19:10 John: I have learned patience.\nD19:11 Tim: Reading can help pass the time.\nD19:12 John: I love non-fiction books about personal development.\nD19:13 John: I recently finished rereading The Alchemist.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        let draft = pipeline.unified_draft.expect("same-source bridge answer");
        assert!(draft.text.contains("The Alchemist"), "{}", draft.text);
        assert!(!draft.text.contains("non-fiction books"), "{}", draft.text);
        assert!(!draft.text.contains("book about castles"), "{}", draft.text);
        assert!(pipeline.evidence_atoms.iter().any(|atom| {
            atom.text.contains("hurt my ankle")
                && atom
                    .supports_slots
                    .contains(&"source_qualifier_anchor".to_string())
        }));
    }

    #[test]
    fn distant_qualifier_bridge_rejects_a_different_speakers_reading() {
        let pipeline = plan_answer_pipeline(
            "What was John reading during his recovery from an ankle injury?",
            &[test_item(
                19,
                0.8,
                "D19:6 John: I hurt my ankle and stayed focused on my recovery.\nD19:7 Tim: I recently finished rereading The Alchemist.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert!(pipeline.unified_draft.is_none(), "{:#?}", pipeline.receipt);
    }

    #[test]
    fn answerability_requires_date_and_fact_to_share_a_memory() {
        let plan = build_query_plan("When did Melanie paint the sunset?", &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 7,
            content: "Session date: 13 October, 2023\nD17:12 Melanie: I painted the sunset.",
            source: "conversation-17",
            entity: Some("Melanie"),
            score: 0.8,
            is_event: false,
        });
        rank_evidence_atoms(&plan, &mut atoms);
        let coverage = assess_slot_coverage(&plan, &atoms, 1, 1);
        assert_eq!(
            coverage
                .iter()
                .find(|slot| slot.slot == "temporal_scope")
                .map(|slot| slot.status.as_str()),
            Some("covered")
        );
        assert_eq!(
            coverage
                .iter()
                .find(|slot| slot.slot == "subject")
                .map(|slot| slot.status.as_str()),
            Some("covered")
        );
    }

    #[test]
    fn unified_answer_binds_actions_to_the_subject_speaker() {
        let plan = build_query_plan("What did Melanie say about the dinosaur exhibit?", &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 9,
            content: "D6:1 Caroline: Hey Melanie!\nD6:2 Caroline: Thanks, Melanie!\nD6:3 Melanie: They were stoked for the dinosaur exhibit!\nD6:4 Caroline: Wow, Melanie, what a beautiful moment!",
            source: "conversation-6",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        rank_evidence_atoms(&plan, &mut atoms);
        atoms.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        let coverage = assess_slot_coverage(&plan, &atoms, 1, 1);
        let draft = build_unified_answer(
            "What did Melanie say about the dinosaur exhibit?",
            &plan,
            &atoms,
            &coverage,
            None,
            "auto",
        )
        .expect("unified answer");

        assert!(draft.text.contains("dinosaur exhibit"), "{}", draft.text);
        assert!(!draft.text.contains("Hey Melanie"), "{}", draft.text);
        assert!(!draft.text.contains("Thanks, Melanie"), "{}", draft.text);
        assert!(!draft.text.contains("beautiful moment"), "{}", draft.text);
    }

    #[test]
    fn aspect_requests_do_not_require_typed_preference_list_values() {
        let plan = build_query_plan("What does Maya like about Atlas?", &[]);
        assert_eq!(plan.answer_shape, AnswerShape::AspectSet);
        assert!(plan.grounded_list_kind.is_none());
        assert!(!plan.required_slots.iter().any(|slot| slot == "list_value"));

        let list = build_query_plan("What books did Maya read?", &[]);
        assert!(list.grounded_list_kind.is_some());
        assert!(list.required_slots.iter().any(|slot| slot == "list_value"));
    }

    #[test]
    fn typed_location_list_intent_promotes_where_has_to_a_grounded_list() {
        let query = "Where has Melanie camped?";
        let plan = build_query_plan(query, &[]);
        assert_eq!(plan.cardinality, Cardinality::List);
        assert_eq!(plan.answer_shape, AnswerShape::List);
        assert_eq!(
            plan.grounded_list_kind,
            Some(crate::memory_engine_summary::GroundedListKind::Locations)
        );
        assert_eq!(plan.list_scope, "open");
        assert!(plan.required_slots.iter().any(|slot| slot == "list_value"));

        let items = vec![
            test_item(
                4,
                0.30,
                "D4:6 Melanie: I took my family camping in the mountains last week.",
            ),
            test_item(
                6,
                0.20,
                "D6:16 Melanie: Here is a picture of my family camping at the beach.",
            ),
            test_item(
                8,
                0.10,
                "D8:32 Melanie: We went on another camping trip in the forest.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 3, 3, None, "auto", 10, true);
        assert!(
            pipeline.operator_is_admissible(AnswerOperator::GroundedList),
            "{:#?}",
            pipeline.receipt
        );
        for memory_id in [4, 6, 8] {
            assert!(
                pipeline.selected_memory_ids().contains(&memory_id),
                "{:#?}",
                pipeline.receipt
            );
        }
    }

    #[test]
    fn typed_destress_list_bridges_same_turn_activity_and_purpose() {
        let query = "What does Melanie do to destress?";
        let items = vec![
            test_item(
                5,
                0.10,
                "D5:4 Melanie: I just signed up for a pottery class yesterday. It's like therapy for me, letting me express myself and get creative.",
            ),
            test_item(
                7,
                0.30,
                "D7:22 Melanie: I've been running farther to de-stress, which has been great for my headspace.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert!(pipeline.operator_is_admissible(AnswerOperator::GroundedList));
        assert!(
            pipeline.selected_memory_ids().contains(&5),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            pipeline.selected_memory_ids().contains(&7),
            "{:#?}",
            pipeline.receipt
        );
        let rendered_input = pipeline
            .grounded_list_items(&items)
            .into_iter()
            .map(|item| item.content)
            .collect::<Vec<_>>()
            .join("\n");
        for expected in ["pottery", "therapy", "running", "de-stress"] {
            assert!(rendered_input.contains(expected), "{rendered_input}");
        }
    }

    #[test]
    fn document_rerank_keeps_complementary_destress_evidence_admissible() {
        let query = "What does Melanie do to destress?";
        let items = vec![
            test_item(
                2,
                0.99,
                "D2:5 Melanie: I am carving out some me-time each day - running, reading, or playing my violin - which refreshes me.",
            ),
            test_item(
                5,
                0.554_380_9,
                "D5:4 Melanie: I just signed up for a pottery class yesterday. It's like therapy for me, letting me express myself and get creative.",
            ),
            test_item(
                7,
                0.858_539,
                "D7:20 Melanie: Been running longer since our last chat - a great way to destress and clear my mind.",
            ),
            test_item(
                16,
                0.92,
                "D16:8 Melanie: I have found my real muses: painting and pottery. It is so calming and satisfying.",
            ),
        ];

        let reranked = rerank_answer_candidates(query, &items);
        let pipeline = plan_answer_pipeline(query, &reranked, 4, 4, None, "auto", 10, true);
        assert!(
            pipeline.selected_memory_ids().contains(&5),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            pipeline.selected_memory_ids().contains(&7),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !pipeline.selected_memory_ids().contains(&2),
            "weak refresh language must not become a direct decompression fact: {:#?}",
            pipeline.receipt
        );
        assert!(
            !pipeline.selected_memory_ids().contains(&16),
            "calming alone must not answer an exact destress query: {:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn singular_shared_event_does_not_promote_a_generic_event_extractor_to_a_list() {
        let plan = build_query_plan(
            "What did Anthony and John end up playing during the charity event?",
            &[],
        );
        assert_eq!(plan.cardinality, Cardinality::Single);
        assert!(plan.grounded_list_kind.is_none());
    }

    #[test]
    fn lexical_action_forms_bridge_past_tense_queries_to_ongoing_source_actions() {
        assert!(text_matches_term(
            "melanie went camping in the forest",
            "camped"
        ));
        assert!(text_matches_term(
            "caroline is moving to a new city",
            "moved"
        ));
        assert!(text_matches_term("maya is studying engineering", "studied"));
    }

    #[test]
    fn list_quantity_requirement_excludes_calendar_numbers() {
        let numbered = build_query_plan("List 3 projects Maya finished.", &[]);
        assert!(numbered
            .required_slots
            .iter()
            .any(|slot| slot == "quantity"));
        let dated = build_query_plan("List projects Maya finished before 2023-05-07.", &[]);
        assert!(!dated.required_slots.iter().any(|slot| slot == "quantity"));
    }

    #[test]
    fn strategy_backoff_rejects_wrong_speaker_and_unrelated_object() {
        let query = "What does John do to supplement his basketball training?";
        let items = vec![
            test_item(
                1,
                0.8,
                "D1:1 John: I am trying yoga for extra strength alongside basketball training.",
            ),
            test_item(
                2,
                0.99,
                "D2:1 Tim: I use swimming alongside basketball training.",
            ),
            test_item(
                3,
                0.99,
                "D3:1 John: I use painting alongside music lessons.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 3, 3, None, "auto", 10, true);
        assert!(pipeline.selected_memory_ids().contains(&1));
        for atom in &pipeline.evidence_atoms {
            if atom.memory_id != 1 {
                assert!(!atom_is_answer_bearing(atom), "{atom:?}");
            }
        }
    }

    #[test]
    fn typed_question_plan_preserves_all_required_slot_values() {
        let plan = build_query_plan(
            "List all 3 projects that Melanie and Caroline did not finish before 2023-05-07, and infer what was likely if delayed.",
            &[],
        );

        assert_eq!(plan.people, vec!["Melanie", "Caroline"]);
        assert_eq!(plan.temporal_scope, "bounded");
        assert!(plan.time_terms.contains(&"2023-05-07".to_string()));
        assert!(plan.action_terms.iter().any(|term| term == "finish"));
        assert!(plan.quantity_terms.contains(&"3".to_string()));
        assert_eq!(plan.list_scope, "exhaustive");
        assert_eq!(plan.inference_type, InferenceType::Likely);
        assert!(!plan.negation_terms.is_empty());
        assert!(!plan.condition_terms.is_empty());
        assert!(plan.protected_terms.contains(&"melanie".to_string()));
        assert!(plan.required_slots.contains(&"action".to_string()));
        assert!(plan.required_slots.contains(&"quantity".to_string()));
        assert!(plan.required_slots.contains(&"list_scope".to_string()));
        assert!(plan.required_slots.contains(&"negation".to_string()));
        assert!(plan.required_slots.contains(&"condition".to_string()));
    }

    struct ScorerGuard;
    impl Drop for ScorerGuard {
        fn drop(&mut self) {
            crate::semantic_scorer::set_test_scorer(None);
        }
    }

    fn with_table_scorer(
        relevance: Vec<(&'static str, f32)>,
        contradictions: Vec<(&'static str, &'static str)>,
    ) -> ScorerGuard {
        crate::semantic_scorer::set_test_scorer(Some(std::sync::Arc::new(
            crate::semantic_scorer::TableScorer { relevance, contradictions },
        )));
        ScorerGuard
    }

    const ROAD_TRIP_SESSION: &str = "Session date: 5:20 pm on 5 November, 2022\nD30:1 James: Hey John, hope you're doing well. Yesterday, we started on a road trip. It was fun spending time with the family and my dogs.\nD30:2 John: Hey James! Wow, what an adventure!";




    /// Dev-only diagnosis: is the gold turn reranked, and how is it scored?
    /// `LOCOMO_DATASET=... cargo test --release --lib rerank_coverage_diagnosis -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn rerank_coverage_diagnosis() {
        // Per-question funnel records for the fixed dev sample (first N
        // questions per category per conversation), written as JSON lines.
        const DEV: &[&str] = &["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"];
        let per_category = std::env::var("DIAGNOSIS_PER_CATEGORY").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(6);
        let out_path = std::env::var("DIAGNOSIS_OUT").unwrap_or_else(|_| "target/eval/funnel.jsonl".into());
        let data: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap(),
        )
        .unwrap();
        let root = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/tidemark/models");
        let scorer = crate::semantic_scorer::OnnxSemanticScorer::load(
            &root.join("bge-reranker-v2-m3"),
            &root.join("mdeberta-v3-base-xnli"),
            6,
        )
        .unwrap();
        let mut out = std::fs::File::create(&out_path).unwrap();
        for sample in data.as_array().unwrap() {
            let id = sample["sample_id"].as_str().unwrap();
            if !DEV.contains(&id) {
                continue;
            }
            let conversation = sample["conversation"].as_object().unwrap();
            let mut items = Vec::new();
            for (key, turns) in conversation {
                let Some(number) = key.strip_prefix("session_").and_then(|rest| rest.parse::<u64>().ok()) else { continue };
                let date = conversation[&format!("session_{number}_date_time")].as_str().unwrap_or("");
                let mut content = format!("Session date: {date}\n");
                for turn in turns.as_array().unwrap() {
                    content.push_str(&format!(
                        "{} {}: {}\n",
                        turn["dia_id"].as_str().unwrap_or(""),
                        turn["speaker"].as_str().unwrap_or(""),
                        turn["text"].as_str().unwrap_or("")
                    ));
                }
                items.push(test_item(number, 0.5, &content));
            }
            let mut counts = HashMap::<i64, usize>::new();
            for (index, qa) in sample["qa"].as_array().unwrap().iter().enumerate() {
                let category = qa["category"].as_i64().unwrap_or(0);
                let seen = counts.entry(category).or_default();
                if *seen >= per_category {
                    continue;
                }
                *seen += 1;
                if category == 5 {
                    continue;
                }
                let gold = qa["evidence"].as_array().map(|v| v.iter().filter_map(|x| x.as_str().map(str::to_string)).collect::<Vec<_>>()).unwrap_or_default();
                let question = qa["question"].as_str().unwrap();
                let plan = build_query_plan(planner_question_text(question), &items);
                let mut atoms = items
                    .iter()
                    .flat_map(|item| {
                        project_source(ProjectionInput {
                            memory_id: item.id,
                            content: &item.content,
                            source: &item.source,
                            entity: item.entity.as_deref(),
                            score: item.score,
                            is_event: false,
                        })
                    })
                    .collect::<Vec<_>>();
                rank_evidence_atoms(&plan, &mut atoms);
                apply_semantic_relevance(question, &plan, &mut atoms, Some(&scorer));
                let mut ranked = atoms.iter().filter(|a| a.kind.grounding_role() == GroundingRole::Original).collect::<Vec<_>>();
                ranked.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
                let gold_atoms = ranked
                    .iter()
                    .enumerate()
                    .filter(|(_, atom)| atom.source_anchor.as_ref().is_some_and(|anchor| gold.contains(&anchor.value)))
                    .map(|(rank, atom)| serde_json::json!({
                        "anchor": atom.source_anchor.as_ref().map(|a| a.value.clone()),
                        "text": atom.text, "relevance": atom.semantic_relevance, "rank": rank,
                        "answer_bearing": atom_is_answer_bearing(atom),
                        "subjects": atom.fact_subjects, "speaker": atom.speaker,
                    }))
                    .collect::<Vec<_>>();
                let top = ranked.iter().take(5).map(|atom| serde_json::json!({
                    "anchor": atom.source_anchor.as_ref().map(|a| a.value.clone()),
                    "text": atom.text, "relevance": atom.semantic_relevance,
                })).collect::<Vec<_>>();
                let record = serde_json::json!({
                    "key": format!("{id}:{index}"), "category": category, "question": question,
                    "gold_answer": qa["answer"], "gold_turns": gold, "plan_subject": plan.subject,
                    "plan_target": format!("{:?}", plan.answer_target), "gold_atoms": gold_atoms, "top": top,
                });
                use std::io::Write;
                writeln!(out, "{record}").unwrap();
            }
        }
        eprintln!("funnel written to {out_path}");
    }


    #[test]
    fn confident_relevance_covers_qualifiers_and_answer_class() {
        let yoga = "Session date: 4:10 pm on 2 October, 2023\nD19:5 John: My ankle injury is finally healed, back on the court soon.\nD20:1 Tim: I finished my book draft!\nD20:2 John: Hi Tim! Congrats on your success! I'm also trying out yoga to get a little extra strength and flexibility. It's challenging but worth it.";
        let music = "Session date: 3:19 pm on 28 August, 2023\nD15:27 Caroline: What music do you like?\nD15:28 Melanie: I'm a fan of both classical like Bach and Mozart, as well as modern music like Ed Sheeran's \"Perfect\".";
        let run = |question: &str, content: &str, needle: &'static str, relevance: f32| {
            let _scorer = with_table_scorer(vec![(needle, relevance)], Vec::new());
            let mut pipeline = plan_answer_pipeline(question, &[test_item(5, 0.9, content)], 1, 1, None, "auto", 10, true);
            pipeline.set_selected_operator(Some("unified_answer_generator_v1"));
            pipeline.unified_draft.clone().map(|draft| {
                let verified = pipeline
                    .unified_draft_coverage()
                    .is_some_and(|reports| !reports.is_empty() && reports.iter().all(|report| report.covered));
                (draft.draft_mode, verified)
            })
        };
        let question = "What is John trying out to improve his strength and flexibility after recovery from ankle injury?";
        assert_eq!(run(question, yoga, "trying out yoga", 0.97), Some(("synthesized", true)));
        assert!(
            run(question, yoga, "trying out yoga", 0.2).is_none_or(|(mode, verified)| mode != "synthesized" || !verified),
            "without a confident verdict the qualifier must occur lexically"
        );
        let classical = run("Which classical musicians does Melanie enjoy listening to?", music, "Bach and Mozart", 0.97);
        assert_eq!(classical, Some(("partial", true)), "class covered; list completeness still unknown");
    }

    /// Dev-only count: statements whose fact subject is not a conversation
    /// participant (a sentence-initial word mistaken for a name).
    #[test]
    #[ignore]
    fn phantom_subject_census() {
        const DEV: &[&str] = &["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"];
        let data: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap(),
        )
        .unwrap();
        let (mut statements, mut phantom) = (0, 0);
        let mut words = std::collections::BTreeMap::<String, usize>::new();
        for sample in data.as_array().unwrap() {
            if !DEV.contains(&sample["sample_id"].as_str().unwrap()) {
                continue;
            }
            let conversation = sample["conversation"].as_object().unwrap();
            let speakers = [conversation["speaker_a"].as_str().unwrap(), conversation["speaker_b"].as_str().unwrap()];
            for (key, turns) in conversation {
                if !key.starts_with("session_") || key.ends_with("date_time") {
                    continue;
                }
                for turn in turns.as_array().unwrap() {
                    let content = format!("{} {}: {}", turn["dia_id"].as_str().unwrap(), turn["speaker"].as_str().unwrap(), turn["text"].as_str().unwrap());
                    for atom in proof_test_atoms(1, &content) {
                        if atom.kind != EvidenceKind::Statement {
                            continue;
                        }
                        statements += 1;
                        let unknown = atom
                            .fact_subjects
                            .iter()
                            .filter(|subject| !speakers.iter().any(|speaker| normalize_person_key(speaker) == normalize_person_key(subject)))
                            .collect::<Vec<_>>();
                        let speaker_lost = atom.speaker.as_ref().is_some_and(|speaker| !atom.fact_subjects.contains(speaker));
                        if !unknown.is_empty() && speaker_lost && atom.text.starts_with(unknown[0].as_str()) {
                            phantom += 1;
                            *words.entry(unknown[0].clone()).or_default() += 1;
                        }
                    }
                }
            }
        }
        let mut top = words.into_iter().collect::<Vec<_>>();
        top.sort_by(|a, b| b.1.cmp(&a.1));
        eprintln!("{statements} statements; {phantom} lost the speaker to a sentence-initial non-participant 'name'");
        eprintln!("top: {:?}", &top[..top.len().min(40)]);
    }

    #[test]
    fn only_record_attested_names_open_sentences_as_people() {
        let content = "D1:1 Caroline: Keep up the great work!\nD1:2 Melanie: Got a new puppy last week.\nD1:3 Caroline: Jolene told me about it. I met Jolene at the park.\nD1:4 Melanie: Family is everything to me.";
        let atoms = proof_test_atoms(3, content);
        let subjects = |needle: &str| {
            atoms.iter().find(|atom| atom.text.contains(needle)).unwrap().fact_subjects.clone()
        };
        assert_eq!(subjects("Got a new puppy"), vec!["Melanie".to_string()]);
        assert!(!subjects("Keep up").contains(&"Keep".to_string()));
        assert!(!subjects("Family is").contains(&"Family".to_string()));
        assert!(subjects("Jolene told").contains(&"Jolene".to_string()), "attested mid-sentence elsewhere");
        let coordinated = proof_test_atoms(4, "D4:8 John: Anthony and I went to a charity thing.\nD4:9 Tim: Nate's dog is cute.");
        assert!(coordinated[0].mentioned_people.contains(&"Anthony".to_string()) || coordinated.iter().any(|atom| atom.mentioned_people.contains(&"Anthony".to_string())));
        assert!(coordinated.iter().any(|atom| atom.possessive_owners.contains(&"Nate".to_string())));
    }

    #[test]
    fn reactions_and_second_person_statements_are_not_the_speakers_facts() {
        for statement in [
            "That must've been breathtaking!",
            "Glad you got some R&R after the drive.",
            "That sounds like an amazing trip.",
            "Your painting is gorgeous.",
        ] {
            let attribution = fact_attribution(statement, Some("Caroline"));
            assert!(attribution.fact_subjects.is_empty(), "{statement}: {:?}", attribution.fact_subjects);
        }
        let attested = HashSet::from(["caroline".to_string(), "melanie".to_string()]);
        for statement in [
            "Went hiking with the kids last weekend.",
            "Got a new puppy, she's adorable.",
            "Thanks to you, I finally got the job.",
            "That was my favorite trip ever.",
        ] {
            let attribution = fact_attribution_with_names(statement, Some("Caroline"), Some(&attested));
            assert_eq!(attribution.fact_subjects, vec!["Caroline".to_string()], "{statement}");
        }
    }

    /// Real reranker scores: `SLOW_QUESTION=... SLOW_PASSAGES='a|||b' cargo test --release reranker_score_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn reranker_score_probe() {
        let root = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/tidemark/models");
        let scorer = crate::semantic_scorer::OnnxSemanticScorer::load(&root.join("bge-reranker-v2-m3"), &root.join("mdeberta-v3-base-xnli"), 6).unwrap();
        let question = std::env::var("SLOW_QUESTION").unwrap();
        let passages = std::env::var("SLOW_PASSAGES").unwrap().replace("\\n", "\n");
        let passages = passages.split("|||").collect::<Vec<_>>();
        for (passage, score) in passages.iter().zip(crate::semantic_scorer::SemanticScorer::relevance(&scorer, &question, &passages).unwrap()) {
            eprintln!("SCORE {score:.3} {passage:?}");
        }
    }

    /// Census of the atoms the model path can send: `LOCOMO_DATASET=... SLOW_SAMPLE=conv-26 cargo test reasoning_atom_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn reasoning_atom_census() {
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap()).unwrap();
        let sample_id = std::env::var("SLOW_SAMPLE").unwrap();
        let sample = data.as_array().unwrap().iter().find(|s| s["sample_id"] == sample_id.as_str()).unwrap();
        let conversation = sample["conversation"].as_object().unwrap();
        let mut items = Vec::new();
        for (key, turns) in conversation {
            let Some(number) = key.strip_prefix("session_").and_then(|rest| rest.parse::<u64>().ok()) else { continue };
            let date = conversation[&format!("session_{number}_date_time")].as_str().unwrap_or("");
            let mut content = format!("Session date: {date}\n");
            for turn in turns.as_array().unwrap() {
                content.push_str(&format!("{} {}: {}\n", turn["dia_id"].as_str().unwrap_or(""), turn["speaker"].as_str().unwrap_or(""), turn["text"].as_str().unwrap_or("")));
            }
            items.push(test_item(number, 0.5, &content));
        }
        let atoms = reasoning_source_atoms(&items).unwrap();
        let mut kinds = std::collections::BTreeMap::<String, (usize, usize)>::new();
        for atom in atoms.values() {
            let entry = kinds.entry(atom.evidence_kind.clone()).or_default();
            entry.0 += 1;
            entry.1 += atom.body.len();
        }
        eprintln!("CENSUS atoms={} kinds={kinds:?}", atoms.len());
        for atom in atoms.values().take(12) {
            eprintln!("CENSUS sample {} {} {:?} {:?}", atom.atom_id, atom.evidence_kind, atom.source_anchor.as_ref().map(|a| a.value.clone()), atom.body.chars().take(90).collect::<String>());
        }
    }

    /// Latency harness: `UMEM_PIPELINE_TRACE=1 SLOW_SAMPLE=conv-47 SLOW_QUESTION="..." cargo test --release ... pipeline_latency_probe -- --ignored --nocapture`


    /// Rerank-preselection coverage (dev): share of gold turns that reach the
    /// reranker under different preselection strategies. No reranker calls.
    #[test]
    #[ignore]
    fn preselection_coverage() {
        const DEV: &[&str] = &["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"];
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap()).unwrap();
        let root = std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/tidemark/models");
        let embedder = crate::semantic_scorer::OnnxEmbedder::load(&root.join("multilingual-e5-small"), 6).unwrap();
        let strategies: [(&str, usize, usize); 6] = [("lexical48", 48, 0), ("lexical96", 96, 0), ("lex24+dense24", 24, 24), ("lex48+dense24", 48, 24), ("lex48+dense48", 48, 48), ("dense48", 0, 48)];
        let mut covered = vec![0usize; strategies.len()];
        let mut total = 0usize;
        for sample in data.as_array().unwrap() {
            let id = sample["sample_id"].as_str().unwrap();
            if !DEV.contains(&id) { continue; }
            let conversation = sample["conversation"].as_object().unwrap();
            let mut items = Vec::new();
            for (key, turns) in conversation {
                let Some(number) = key.strip_prefix("session_").and_then(|rest| rest.parse::<u64>().ok()) else { continue };
                let date = conversation[&format!("session_{number}_date_time")].as_str().unwrap_or("");
                let mut content = format!("Session date: {date}\n");
                for turn in turns.as_array().unwrap() {
                    content.push_str(&format!("{} {}: {}\n", turn["dia_id"].as_str().unwrap_or(""), turn["speaker"].as_str().unwrap_or(""), turn["text"].as_str().unwrap_or("")));
                }
                items.push(test_item(number, 0.5, &content));
            }
            // Turn texts (with the preceding turn) for dense scoring, keyed like the pipeline.
            let base_atoms = items.iter().flat_map(|item| project_source(ProjectionInput {
                memory_id: item.id, content: &item.content, source: &item.source,
                entity: item.entity.as_deref(), score: item.score, is_event: false,
            })).collect::<Vec<_>>();
            let mut turn_text = BTreeMap::<(u64, usize), (String, String)>::new();
            for atom in base_atoms.iter().filter(|a| a.kind == EvidenceKind::Statement && a.context_group != usize::MAX) {
                let entry = turn_text.entry((atom.memory_id, atom.context_group)).or_insert_with(|| (atom.speaker.clone().unwrap_or_default(), String::new()));
                if !entry.1.is_empty() { entry.1.push(' '); }
                entry.1.push_str(&atom.text);
            }
            let keys = turn_text.keys().copied().collect::<Vec<_>>();
            let passages = keys.iter().map(|key| {
                let (speaker, body) = &turn_text[key];
                let previous = turn_text.get(&(key.0, key.1.wrapping_sub(1))).map(|(s, b)| format!("{s}: {b}\n")).unwrap_or_default();
                format!("{previous}{speaker}: {body}")
            }).collect::<Vec<_>>();
            let passage_vectors = embedder.embed_passages(&passages.iter().map(String::as_str).collect::<Vec<_>>()).unwrap();
            let anchor_of = base_atoms.iter().filter_map(|a| a.source_anchor.as_ref().map(|anchor| (anchor.value.clone(), (a.memory_id, a.context_group)))).collect::<HashMap<_, _>>();
            for qa in sample["qa"].as_array().unwrap() {
                if qa["category"].as_i64() == Some(5) { continue; }
                let gold = qa["evidence"].as_array().map(|v| v.iter().filter_map(|x| x.as_str()).filter_map(|id| anchor_of.get(id).copied()).collect::<HashSet<_>>()).unwrap_or_default();
                if gold.is_empty() { continue; }
                let question = qa["question"].as_str().unwrap();
                let plan = build_query_plan(planner_question_text(question), &items);
                let mut atoms = base_atoms.clone();
                rank_evidence_atoms(&plan, &mut atoms);
                let mut lexical = HashMap::<(u64, usize), f32>::new();
                for atom in atoms.iter().filter(|a| a.kind == EvidenceKind::Statement && a.context_group != usize::MAX) {
                    let best = lexical.entry((atom.memory_id, atom.context_group)).or_insert(f32::NEG_INFINITY);
                    *best = best.max(atom.score);
                }
                let mut lexical_ranked = lexical.into_iter().collect::<Vec<_>>();
                lexical_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                let query = embedder.embed_queries(&[question]).unwrap().remove(0);
                let mut dense_ranked = keys.iter().zip(&passage_vectors).map(|(key, vector)| (*key, crate::semantic_scorer::dot(&query, vector))).collect::<Vec<_>>();
                dense_ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
                total += gold.len();
                for (index, (_, lexical_n, dense_n)) in strategies.iter().enumerate() {
                    let mut chosen = lexical_ranked.iter().take(*lexical_n).map(|(key, _)| *key).collect::<HashSet<_>>();
                    let mut added = 0;
                    for (key, _) in &dense_ranked {
                        if added >= *dense_n { break; }
                        if chosen.insert(*key) { added += 1; }
                    }
                    covered[index] += gold.iter().filter(|key| chosen.contains(key)).count();
                }
            }
        }
        for ((name, _, _), count) in strategies.iter().zip(&covered) {
            eprintln!("COVERAGE {name:16} {count}/{total} = {:.1}%", 100.0 * *count as f32 / total as f32);
        }
    }

    /// Planner census over all LoCoMo questions (answer typing vs. gold).
    /// `LOCOMO_DATASET=... cargo test --lib planner_census -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn planner_census() {
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap()).unwrap();
        let mut temporal = (0, 0, Vec::new());
        let mut lists = (0, 0, Vec::new());
        for sample in data.as_array().unwrap() {
            let c = sample["conversation"].as_object().unwrap();
            let speakers = [c["speaker_a"].as_str().unwrap(), c["speaker_b"].as_str().unwrap()];
            let items = vec![test_item(1, 0.5, &format!("D1:1 {}: hi\nD1:2 {}: hello", speakers[0], speakers[1]))];
            for qa in sample["qa"].as_array().unwrap() {
                if qa["category"].as_i64() == Some(5) { continue; }
                let q = qa["question"].as_str().unwrap();
                let gold = qa["answer"].to_string();
                let plan = build_query_plan(planner_question_text(q), &items);
                if qa["category"].as_i64() == Some(2) {
                    temporal.0 += 1;
                    if plan.answer_target != AnswerTarget::Temporal {
                        temporal.1 += 1;
                        if temporal.2.len() < 15 { temporal.2.push(format!("{q} -> {:?} | gold {gold}", plan.answer_target)); }
                    }
                }
                let gold_items = gold.split(',').filter(|part| !part.trim().is_empty()).count();
                if gold_items >= 2 && !q.to_lowercase().starts_with("when") && !q.to_lowercase().starts_with("how") {
                    lists.0 += 1;
                    if plan.cardinality != Cardinality::List {
                        lists.1 += 1;
                        if lists.2.len() < 20 { lists.2.push(format!("{q} -> {:?} | gold {gold}", plan.cardinality)); }
                    }
                }
            }
        }
        eprintln!("CENSUS temporal (cat 2) not planned temporal: {}/{}", temporal.1, temporal.0);
        for e in &temporal.2 { eprintln!("CENSUS   T {e}"); }
        eprintln!("CENSUS multi-item gold not planned as list: {}/{}", lists.1, lists.0);
        for e in &lists.2 { eprintln!("CENSUS   L {e}"); }
    }

    #[test]
    #[ignore]
    fn tmp_subject_census() {
        let data: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap()).unwrap();
        let (mut total, mut wrong) = (0, 0);
        let mut examples = Vec::new();
        for sample in data.as_array().unwrap() {
            let c = sample["conversation"].as_object().unwrap();
            let speakers = [c["speaker_a"].as_str().unwrap(), c["speaker_b"].as_str().unwrap()];
            let items = vec![test_item(1, 0.5, &format!("D1:1 {}: hi\nD1:2 {}: hello", speakers[0], speakers[1]))];
            for qa in sample["qa"].as_array().unwrap() {
                let q = qa["question"].as_str().unwrap();
                let named = speakers.iter().find(|s| q.contains(*s));
                let Some(named) = named else { continue };
                total += 1;
                let plan = build_query_plan(planner_question_text(q), &items);
                let ok = plan.subject.as_deref().is_some_and(|subject| speakers.iter().any(|s| normalize_person_key(s) == normalize_person_key(subject)));
                if !ok { wrong += 1; if examples.len() < 12 { examples.push(format!("{q} -> {:?} (named {named})", plan.subject)); } }
            }
        }
        eprintln!("CENSUS {wrong}/{total} questions naming a participant get a non-participant subject");
        for e in examples { eprintln!("CENSUS   {e}"); }
    }



    #[test]
    fn plural_answer_questions_are_lists() {
        for question in [
            "What are Joanna's hobbies?",
            "What were Melanie's favorite books?",
            "What people has Maria met and helped while volunteering?",
            "What books has Tim read?",
            "What types of pottery have Melanie and her kids made?",
        ] {
            assert_eq!(build_query_plan(question, &[]).cardinality, Cardinality::List, "{question}");
        }
        for question in [
            "What is Joanna's favorite movie?",
            "What is Caroline's relationship status?",
            "What kind of flooring is Gina looking for in her dance studio?",
        ] {
            assert_eq!(build_query_plan(question, &[]).cardinality, Cardinality::Single, "{question}");
        }
    }

    #[test]
    fn confidently_relevant_relative_date_answers_the_year_question() {
        let session = "Session date: 4:21 pm on 16 July, 2023\nD3:26 Tim: How long have you been surfing?\nD3:27 John: I started surfing five years ago and it's been great.";
        let _scorer = with_table_scorer(vec![("started surfing five years ago", 0.87)], Vec::new());
        let mut pipeline = plan_answer_pipeline("What year did John start surfing?", &[test_item(3, 0.9, session)], 1, 1, None, "auto", 10, true);
        pipeline.set_selected_operator(Some("unified_answer_generator_v1"));
        let draft = pipeline.unified_draft.clone().expect("date draft");
        assert!(draft.text.contains("2018"), "{}", draft.text);
        assert!(!pipeline.verify_claim_evidence("The date was 2018.", &HashSet::from([3])).memory_ids.is_empty());
    }

    #[test]
    fn questions_about_what_was_suggested_are_not_recommendation_requests() {
        for question in ["What did Andrew suggest as a way to reduce carbon footprint?",
            "How does Andrew suggest helping the planet while also training the body?",
            "What things has Nate reccomended to Joanna?", "Which book was recommended to Melanie?"] {
            assert!(!query_requests_preference_recommendation(&question.to_lowercase()), "{question}");
        }
        for question in ["What hobby would you recommend for Caroline?", "Can you suggest a good hobby for Jon?", "Which hobby should Maria pick up?"] {
            assert!(query_requests_preference_recommendation(&question.to_lowercase()), "{question}");
        }
    }

    #[test]
    fn year_and_month_questions_are_temporal() {
        for question in ["What year did John start surfing?", "In which month did Maria move?", "Which year did Tim visit the UK?", "In which month's game did John achieve a career-high score in points?"] {
            assert_eq!(build_query_plan(question, &[]).answer_target, AnswerTarget::Temporal, "{question}");
        }
    }

    #[test]
    fn uninverted_questions_keep_the_named_subject() {
        for (question, subject) in [
            ("When Jon has lost his job as a banker?", "Jon"),
            ("When Gina has lost her job at Door Dash?", "Gina"),
            ("When did Jon lose his job?", "Jon"),
        ] {
            assert_eq!(build_query_plan(question, &[]).subject.as_deref(), Some(subject), "{question}");
        }
    }

    #[test]
    fn within_confident_relevance_the_more_relevant_sentence_is_selected() {
        let question = "Why did Caroline choose the adoption agency?";
        let items = [test_item(2, 0.9, "Session date: 1:14 pm on 25 May, 2023\nD2:7 Melanie: My kids are so excited about summer break! Any fun plans for the summer?\nD2:8 Caroline: Researching adoption agencies — it's been a dream to have a family and give a loving home to kids who need it.\nD2:11 Melanie: Wow, that agency looks great! What made you pick it?\nD2:12 Caroline: I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me.")];
        let _scorer = with_table_scorer(vec![("Researching adoption agencies", 0.85), ("help LGBTQ+ folks", 0.997), ("Their inclusivity", 0.74)], Vec::new());
        let pipeline = plan_answer_pipeline(question, &items, 1, 1, None, "auto", 10, true);
        let selected = pipeline.receipt.evidence_plan.selected_answer_atom_ids.clone();
        let first = pipeline.evidence_atoms.iter().find(|atom| atom.atom_id == selected[0]).unwrap();
        assert!(first.text.contains("help LGBTQ+ folks"), "{selected:?} {}", first.text);
    }

    #[test]
    fn a_reaction_cannot_borrow_relevance_from_the_previous_turn() {
        let question = "How did Caroline feel while watching the meteor shower?";
        let plan = build_query_plan(question, &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 4,
            content: "Session date: 1:36 pm on 3 July, 2023\nD9:12 Melanie: I'll always remember watching the Perseid meteor shower, I felt tiny and in awe of the universe.\nD9:13 Caroline: That must've been breathtaking!",
            source: "dialogue",
            entity: Some("Caroline"),
            score: 0.9,
            is_event: false,
        });
        rank_evidence_atoms(&plan, &mut atoms);
        // With the previous turn as context the reaction looks relevant; on
        // its own it says nothing about the meteor shower.
        let scorer = crate::semantic_scorer::TableScorer {
            relevance: vec![("context:Perseid meteor shower", 0.9), ("That must've been", 0.001)],
            contradictions: Vec::new(),
        };
        apply_semantic_relevance(question, &plan, &mut atoms, Some(&scorer));
        let reaction = atoms.iter().find(|atom| atom.text.contains("breathtaking")).unwrap();
        let score = reaction.semantic_relevance.unwrap();
        assert!(score < crate::semantic_scorer::grant_threshold(), "{score}");
        assert!(!reaction.supports_slots.iter().any(|slot| slot == "semantic_relevance"));
    }

    #[test]
    fn relevant_turn_grants_only_its_answer_bearing_sentence() {
        let question = "When did Jon lose his job as a banker?";
        let plan = build_query_plan(question, &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 3,
            content: "Session date: 4:04 pm on 20 January, 2023\nD1:1 Gina: How have you been?\nD1:2 Jon: Sorry to hear that! I lost my job as a banker yesterday. I get it, Gina.",
            source: "dialogue",
            entity: Some("Jon"),
            score: 0.9,
            is_event: false,
        });
        rank_evidence_atoms(&plan, &mut atoms);
        let scorer = crate::semantic_scorer::TableScorer {
            relevance: vec![("lost my job as a banker", 0.95)],
            contradictions: Vec::new(),
        };
        let receipt = apply_semantic_relevance(question, &plan, &mut atoms, Some(&scorer));
        assert_eq!(receipt.map(|(backend, _)| backend).as_deref(), Some("test:table"));
        let find = |needle: &str| atoms.iter().find(|atom| atom.text.contains(needle)).unwrap();
        let answer = find("lost my job");
        assert_eq!(answer.semantic_relevance, Some(0.95));
        assert!(answer.supports_slots.iter().any(|slot| slot == "semantic_relevance"));
        for filler in ["Sorry to hear that", "I get it"] {
            let atom = find(filler);
            assert_eq!(atom.semantic_relevance, Some(0.2), "{filler} is judged on its own");
            assert!(!atom.supports_slots.iter().any(|slot| slot == "semantic_relevance"), "{filler}");
        }
    }


    #[test]
    fn general_questions_need_confident_relevance_but_dates_and_renderers_do_not() {
        let session = "D17:12 Melanie: I painted a sunset.";
        let question = "What did Melanie paint?";
        let verified = |operator: Option<&str>| {
            let mut pipeline = plan_answer_pipeline(question, &[test_item(5, 0.9, session)], 1, 1, None, "auto", 10, true);
            pipeline.set_selected_operator(operator);
            let draft = pipeline.unified_draft.clone().expect("draft");
            !pipeline.verify_claim_evidence(&draft.text, &HashSet::from([5])).memory_ids.is_empty()
        };
        assert!(verified(Some("unified_answer_generator_v1")), "lexical mode");
        {
            let _scorer = with_table_scorer(vec![("painted a sunset", 0.2)], Vec::new());
            assert!(!verified(Some("unified_answer_generator_v1")), "uncertain relevance is not enough");
            assert!(verified(None), "specialized renderers own their relevance");
        }
        {
            let _scorer = with_table_scorer(vec![("painted a sunset", 0.9)], Vec::new());
            assert!(verified(Some("unified_answer_generator_v1")));
        }
        // Dates keep their arithmetic proof in the uncertain band.
        let _scorer = with_table_scorer(vec![("road trip", 0.2)], Vec::new());
        let mut pipeline = plan_answer_pipeline(
            "When did James start the road trip?",
            &[test_item(180, 0.9, ROAD_TRIP_SESSION)],
            1, 1, None, "auto", 10, true,
        );
        pipeline.set_selected_operator(Some("unified_answer_generator_v1"));
        assert!(!pipeline
            .verify_claim_evidence("The date was 4 November 2022.", &HashSet::from([180]))
            .memory_ids
            .is_empty());
    }

    #[test]
    fn semantic_contradiction_vetoes_a_bound_claim() {
        let plan = build_query_plan("What does James think about cooking?", &[]);
        let atoms = proof_test_atoms(9, "D9:1 James: I never liked cooking.");
        let ids = atoms
            .iter()
            .filter(|atom| atom.text.contains("cooking"))
            .map(|atom| atom.atom_id.clone())
            .collect::<Vec<_>>();
        assert!(!semantic_contradiction_veto(&plan, &atoms, &ids, "James loves cooking."));
        let _scorer = with_table_scorer(Vec::new(), vec![("never liked cooking", "loves cooking")]);
        assert!(semantic_contradiction_veto(&plan, &atoms, &ids, "James loves cooking."));
        assert!(!semantic_contradiction_veto(&plan, &atoms, &ids, "James never liked cooking."));
        let temporal = build_query_plan("When did James stop cooking?", &[]);
        assert!(!semantic_contradiction_veto(&temporal, &atoms, &ids, "James loves cooking."));
    }


    #[test]
    fn slot_reranker_promotes_the_requested_event_over_a_similar_neighbor() {
        let items = vec![
            test_item(
                1,
                0.99,
                "Session date: 20 July 2023\nD10:3 Caroline: I joined a new activist group last week.",
            ),
            test_item(
                2,
                0.10,
                "Session date: 8 May 2023\nD1:3 Caroline: I went to an LGBTQ support group yesterday and felt welcomed.",
            ),
        ];

        let reranked =
            rerank_answer_candidates("When did Caroline go to the LGBTQ support group?", &items);
        assert_eq!(reranked.first().map(|item| item.id), Some(2));
    }

    #[test]
    fn temporal_slot_requires_the_date_and_requested_action_to_share_a_record() {
        let plan = build_query_plan("When did Caroline go to the support group?", &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 1,
            content: "Session date: 8 May 2023",
            source: "date-only",
            entity: Some("Caroline"),
            score: 0.9,
            is_event: false,
        });
        atoms.extend(project_source(ProjectionInput {
            memory_id: 2,
            content: "D1:3 Caroline: I went to the support group yesterday.",
            source: "event-only",
            entity: Some("Caroline"),
            score: 0.8,
            is_event: false,
        }));
        rank_evidence_atoms(&plan, &mut atoms);

        let coverage = assess_slot_coverage(&plan, &atoms, 2, 2);
        assert_eq!(
            coverage
                .iter()
                .find(|slot| slot.slot == "temporal_scope")
                .map(|slot| slot.status.as_str()),
            Some("missing")
        );
    }

    #[test]
    fn temporal_event_selection_keeps_time_on_the_event_sentence() {
        let query = "When did Caroline give a speech at a school?";
        let pipeline = plan_answer_pipeline(
            query,
            &[
                test_item(
                    5,
                    0.99,
                    "Session date: 3 July, 2023\nD5:3 Caroline: Thanks, Mel! I felt powerful giving my talk.",
                ),
                test_item(
                    3,
                    0.10,
                    "Session date: 9 June, 2023\nD3:1 Caroline: I wanted to tell you about my school event last week.\nD3:3 Caroline: I felt powerful giving my talk.",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        let selected_ids = pipeline
            .receipt
            .evidence_plan
            .selected_answer_atom_ids
            .iter()
            .collect::<HashSet<_>>();
        let selected_originals = pipeline
            .evidence_atoms
            .iter()
            .filter(|atom| {
                atom.kind.grounding_role() == GroundingRole::Original
                    && selected_ids.contains(&atom.atom_id)
            })
            .collect::<Vec<_>>();
        assert!(
            selected_originals
                .iter()
                .any(|atom| atom.text.contains("school event last week")),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !selected_originals
                .iter()
                .any(|atom| atom.text.contains("Thanks, Mel")),
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline.unified_draft.expect("school speech date answer");
        assert!(draft.text.contains("2 June 2023"), "{}", draft.text);
        assert!(!draft.text.contains("Thanks, Mel"), "{}", draft.text);
    }

    #[test]
    fn exhaustive_list_scope_is_partial_when_the_candidate_scope_is_incomplete() {
        let plan = build_query_plan("List all projects Melanie completed.", &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 1,
            content: "D1:1 Melanie: I completed the community mural project.",
            source: "project",
            entity: Some("Melanie"),
            score: 0.8,
            is_event: false,
        });
        rank_evidence_atoms(&plan, &mut atoms);

        let coverage = assess_slot_coverage(&plan, &atoms, 10, 1);
        assert_eq!(
            coverage
                .iter()
                .find(|slot| slot.slot == "list_scope")
                .map(|slot| slot.status.as_str()),
            Some("partial")
        );
    }



    #[test]
    fn proper_nouns_do_not_become_people_slots() {
        for query in [
            "What happened in November?",
            "Which European route crosses the Rocky Mountains?",
            "Which Harry Potter book did Melanie read?",
            "What did Nike release?",
        ] {
            let plan = build_query_plan(query, &[]);
            assert!(plan.people.is_empty(), "{query}: {:?}", plan.people);
            assert!(
                !plan
                    .required_slots
                    .iter()
                    .any(|slot| slot.starts_with("person:")),
                "{query}: {:?}",
                plan.required_slots
            );
        }
    }

    #[test]
    fn known_speakers_are_kept_as_distinct_people() {
        let items = vec![test_item(
            1,
            0.8,
            "D1:1 Anthony: I told John.\nD1:2 John: I told Tim.\nD1:3 Tim: I understood.",
        )];
        let plan = build_query_plan("What did Anthony and John tell Tim?", &items);
        assert_eq!(plan.people, vec!["Anthony", "John", "Tim"]);
        assert!(plan.required_slots.contains(&"person:anthony".to_string()));
        assert!(plan.required_slots.contains(&"person:john".to_string()));
        assert!(plan.required_slots.contains(&"person:tim".to_string()));
    }

    #[test]
    fn evidence_plan_selects_the_answer_before_source_limit_truncation() {
        let query = "What did Melanie say about her greenhouse marriage?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.99,
                    "D1:1 Melanie: I went to a professional networking event.",
                ),
                test_item(
                    2,
                    0.10,
                    "D12:4 Melanie: I got married in a greenhouse surrounded by plants.",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 1, true);
        assert_eq!(pipeline.selected_memory_ids(), &[2]);
        assert!(pipeline
            .receipt
            .evidence_plan
            .dropped_memory_ids
            .contains(&1));
    }

    #[test]
    fn unified_answer_keeps_complementary_cross_record_facts() {
        let query = "What does John like about Lebron James?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.8,
                    "D12:20 John: LeBron is the man - love his skills and leadership.",
                ),
                test_item(
                    2,
                    0.8,
                    "D16:12 John: LeBron's moments of determination and heart are incredible.\nD16:13 John: It's why I enjoy watching him play.",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert_eq!(pipeline.selected_memory_ids().len(), 2);
        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("cross-record answer");
        for expected in ["skills", "leadership", "determination", "heart"] {
            assert!(
                draft.text.to_lowercase().contains(expected),
                "{}",
                draft.text
            );
        }
        assert!(
            draft
                .text
                .starts_with("Direct answer: John highlighted these positive aspects:"),
            "{}",
            draft.text
        );
        assert!(!draft.text.contains("LeBron is the man"), "{}", draft.text);
        assert!(draft.text.contains("memory#1"), "{}", draft.text);
        assert!(draft.text.contains("memory#2"), "{}", draft.text);
    }

    #[test]
    fn semantic_term_matching_respects_boundaries_and_action_groups() {
        assert!(!text_matches_term("reaching goals", "go"));
        assert!(!text_matches_term("started learning violin", "join"));
        assert!(text_matches_term("joined a fantasy forum", "join"));
        assert!(text_matches_term("winning a championship", "win"));
        assert!(text_matches_term("lebron's determination", "lebron"));
        assert!(text_matches_term("organizing a camp", "organize"));
        assert!(text_matches_term("achieved a career high", "achieve"));
        assert!(text_matches_term("attended the event", "attend"));
        assert!(text_matches_term(
            "the waves felt super exciting and free-feeling",
            "feel"
        ));
        assert!(text_matches_lexical_term(
            "career-high in assists",
            "career-high"
        ));
        assert!(text_matches_lexical_term(
            "career-high in assists",
            "assist"
        ));
        assert!(!text_matches_lexical_term("achieve goals", "career-high"));
    }

    #[test]
    fn unified_answer_does_not_treat_starting_as_joining() {
        let query = "What forum did Tim join recently?";
        let plan = build_query_plan(query, &[]);
        assert_eq!(plan.object_terms, vec!["forum"]);
        assert!(plan.required_slots.contains(&"object".to_string()));

        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.10,
                    "D1:1 Tim: Last night I joined a fantasy literature forum.",
                ),
                test_item(
                    2,
                    0.99,
                    "D2:1 Tim: Recently started learning the violin.\nD2:2 Caroline: Hey Tim!",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        let draft = pipeline.unified_draft.expect("joined forum answer");

        assert!(
            draft.text.contains("fantasy literature forum"),
            "{}",
            draft.text
        );
        assert!(!draft.text.contains("violin"), "{}", draft.text);
        assert!(!draft.text.contains("Hey Tim"), "{}", draft.text);
    }

    #[test]
    fn object_slot_prefers_career_high_assist_over_generic_achievement() {
        let query = "What career-high assist performance did John achieve?";
        let plan = build_query_plan(query, &[]);
        assert!(plan.object_terms.contains(&"career-high".to_string()));
        assert!(plan.object_terms.contains(&"assist".to_string()));
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.99,
                    "D1:1 John: It feels great to achieve goals after hard work.",
                ),
                test_item(
                    2,
                    0.10,
                    "D2:1 John: I had a career-high in assists last Friday.",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert_eq!(pipeline.selected_memory_ids(), &[2]);
        let draft = pipeline.unified_draft.expect("career-high assist answer");
        assert!(
            draft.text.contains("career-high in assists"),
            "{}",
            draft.text
        );
        assert!(!draft.text.contains("achieve goals"), "{}", draft.text);
    }


    #[test]
    fn descriptive_answer_prefers_exact_object_phrase_and_one_default_node() {
        let query = "How did John describe the team bond?";
        let plan = build_query_plan(query, &[]);
        assert_eq!(plan.answer_target, AnswerTarget::Description);
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.99,
                    "D25:14 John: Our team has grown in communication and bonding.",
                ),
                test_item(
                    2,
                    0.10,
                    "D5:6 John: Our team bond is awesome and worth the hard work.",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert_eq!(pipeline.selected_memory_ids().first(), Some(&2));
        let draft = pipeline.unified_draft.expect("descriptive answer");
        assert!(
            draft.text.contains("team bond is awesome"),
            "{}",
            draft.text
        );
        assert!(
            !draft.text.contains("communication and bonding"),
            "{}",
            draft.text
        );
    }

    #[test]
    fn description_prefers_direct_property_over_related_activity() {
        let query = "How did John describe the team bond?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    22,
                    0.99,
                    "D22:7 Tim: Anything special you do to keep that bond strong?\nD22:8 John: Sounds awesome! Same with us - we have team dinners, outings, and basketball games. It's those moments away from practice that really build and strengthen our unity.",
                ),
                test_item(
                    3,
                    0.20,
                    "D3:7 John: Being surrounded by teammates who are equally passionate creates a strong bond. We push each other to be our best and the love for the game is infectious. It's like having a second family.",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);

        assert_eq!(
            pipeline.selected_memory_ids().first(),
            Some(&3),
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline.unified_draft.expect("direct descriptive answer");
        assert!(draft.text.contains("strong bond"), "{}", draft.text);
        assert!(!draft.text.contains("team dinners"), "{}", draft.text);
    }

    #[test]
    fn count_answer_does_not_extract_an_unrelated_number() {
        let query = "How many times did Melanie injure her ankle?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(1, 0.99, "D1:1 Melanie: I am 25 and play soccer."),
                test_item(2, 0.20, "D4:2 Melanie: I injured my ankle twice."),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        let draft = pipeline.unified_draft.expect("count answer");
        assert!(draft.text.to_lowercase().contains("twice"));
        assert!(!draft.text.contains("25"));
    }

    #[test]
    fn finalized_claims_define_the_only_final_evidence_set() {
        let query = "What did Melanie build?";
        let items = rerank_answer_candidates(
            query,
            &[test_item(2, 0.8, "D2:1 Melanie: I built a greenhouse.")],
        );
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        let receipt = finalize_answer_pipeline_receipt(
            pipeline.receipt,
            &[CoverageReport {
                claim: "Melanie built a greenhouse.".to_string(),
                covered: true,
                evidence_ids: vec![2],
                evidence_atom_ids: vec!["memory:2:atom:0".to_string()],
                evidence_source_anchors: Vec::new(),
                derivation: Vec::new(),
                verification_status: "verified".to_string(),
            }],
            "synthesized",
            "unified_answer_generator_v1",
            false,
            &["unified_answer_generator_v1".to_string()],
            Some("unified_answer_generator_v1"),
        );
        assert_eq!(receipt.final_evidence_ids, vec![2]);
        assert_eq!(receipt.claim_plan[0].evidence_ids, vec![2]);
        assert_eq!(
            receipt.claim_plan[0].text.as_deref(),
            Some("Melanie built a greenhouse.")
        );
    }

    #[test]
    fn fact_attribution_keeps_speaker_address_and_owner_separate() {
        let attribution = fact_attribution(
            "Thanks, Caroline. My kids are excited about the summer break.",
            Some("Melanie"),
        );

        assert_eq!(attribution.fact_subjects, vec!["Melanie"]);
        assert_eq!(attribution.addressed_people, vec!["Caroline"]);
        assert!(!attribution
            .fact_subjects
            .iter()
            .any(|person| normalize_person_key(person) == "caroline"));
        assert!(attribution
            .subject_binding
            .iter()
            .any(|reason| reason.starts_with("speaker_first_person:")));
    }

    #[test]
    fn evidence_chain_uses_two_facts_for_one_location_answer_node() {
        let query = "Where did Caroline move from?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    3,
                    0.99,
                    "D3:13 Caroline: I moved from my home country four years ago.",
                ),
                test_item(
                    4,
                    0.95,
                    "D4:3 Caroline: My home country, Sweden, is important to me.",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, Some(1), "concise", 2, true);
        let selected = pipeline
            .selected_memory_ids()
            .iter()
            .copied()
            .collect::<HashSet<_>>();

        assert_eq!(selected, HashSet::from([3, 4]), "{:#?}", pipeline.receipt);
        assert!(pipeline
            .receipt
            .evidence_plan
            .selected_bridge_atom_ids
            .iter()
            .any(|atom_id| atom_id.starts_with("memory:3:")));
        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("one rendered location answer");
        assert!(draft.text.contains("Sweden"), "{}", draft.text);
        assert!(draft.text.contains("memory#3"), "{}", draft.text);
        assert!(draft.text.contains("memory#4"), "{}", draft.text);
        assert_eq!(
            pipeline.receipt.claim_plan.len(),
            1,
            "{:#?}",
            pipeline.receipt
        );
        assert_eq!(pipeline.receipt.claim_plan[0].evidence_ids, vec![3, 4]);
        assert_eq!(
            pipeline.receipt.claim_plan[0].evidence_source_anchors.len(),
            2,
            "{:#?}",
            pipeline.receipt.claim_plan[0]
        );
    }

    #[test]
    fn quoted_title_does_not_create_a_negation_constraint() {
        assert!(
            extract_negation_terms("When did Melanie read \"Nothing Is Impossible\"?").is_empty()
        );
        assert_eq!(
            extract_negation_terms("What did Melanie not read?"),
            vec!["not"]
        );
    }

    #[test]
    fn final_claim_binding_does_not_expand_to_unrelated_same_record_atoms() {
        let mut atoms = project_source(ProjectionInput {
            memory_id: 7,
            content: "D7:1 John: I built a greenhouse.\nD7:2 John: I painted a shed after work.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        for atom in &mut atoms {
            atom.supports_slots.push("answer_bearing_fact".to_string());
            atom.supports_slots.push("action".to_string());
        }

        let plan = build_query_plan("What did John build?", &[]);
        let (atom_ids, anchors, _) =
            verified_claim_bindings_for_plan(&plan, &atoms, "John built a greenhouse.", &[7]);
        assert_eq!(atom_ids, vec!["memory:7:atom:0"]);
        assert_eq!(anchors.len(), 1);
        assert_eq!(anchors[0].value, "D7:1");
        assert!(
            verified_claim_bindings_for_plan(&plan, &atoms, "John built a shed.", &[7])
                .0
                .is_empty()
        );
    }

    #[test]
    fn universal_profile_summary_claim_bindings_ignore_sibling_modality_logic() {
        let summaries = [
            "Captures how to reuse the profile in future answers, generation, briefings, recommendations, or adaptations without overgeneralizing.",
            "Captures when the profile applies: situations, environments, tasks, collaborators, tools, artifacts, or states that trigger the pattern.",
            "Captures favored choices and avoided patterns when evidence shows a stable preference rather than a one-off event.",
        ];
        let mut atoms = summaries
            .iter()
            .enumerate()
            .flat_map(|(index, summary)| {
                let content = serde_json::json!({
                    "memory_type": "universal_profile",
                    "source_id": "cross-domain-pattern",
                    "subject": "cross-domain recurring pattern",
                    "summary": summary,
                    "evidence_modalities": ["video", "screenshot", "sensor"]
                })
                .to_string();
                project_source(ProjectionInput {
                    memory_id: index as u64 + 2,
                    content: &content,
                    source: "universal-profile-extract",
                    entity: None,
                    score: 0.8,
                    is_event: false,
                })
            })
            .collect::<Vec<_>>();
        let plan = build_query_plan(
            "Explain the universal profile for cross-domain-pattern. This should be a boundless profile extraction answer with recurring pattern, transfer guidance, and evidence modalities.",
            &[],
        );
        rank_evidence_atoms(&plan, &mut atoms);
        for summary in summaries {
            let bindings = verified_claim_bindings_for_plan(&plan, &atoms, summary, &[2, 3, 4]);
            assert!(!bindings.0.is_empty(), "summary={summary}; bindings={bindings:?}");
            assert!(bindings.0.iter().any(|atom_id| atom_id.ends_with(":atom:0")));
            assert!(bindings.0.iter().any(|atom_id| atom_id.ends_with(":atom:1")));
        }
    }

    #[test]
    fn corrected_preference_profile_summary_remains_answer_bearing() {
        let content = serde_json::json!({
            "memory_type": "universal_profile",
            "source_id": "boundaryless-preference-tuning",
            "subject": "user preference adaptation",
            "profile_kind": "generalized_preference_tuning",
            "summary": "Generalized preference tuning adapts runtime behavior from reviewable cross-domain evidence; it does not silently train model weights.",
            "evidence_modalities": ["text", "ui"]
        })
        .to_string();
        let query = "How should generalized preference tuning fine-tune preferences to match user preferences across any domain? Cite the universal profile.";
        let plan = build_query_plan(query, &[]);
        let mut atoms = project_source(ProjectionInput {
            memory_id: 15,
            content: &content,
            source: "/memory/preference-tuning/correct",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        rank_evidence_atoms(&plan, &mut atoms);
        let summary = atoms
            .iter()
            .find(|atom| {
                atom.subject_binding
                    .iter()
                    .any(|binding| binding == "structured_profile_summary")
            })
            .expect("corrected summary atom");
        assert!(
            summary
                .supports_slots
                .iter()
                .any(|slot| slot == "answer_bearing_fact"),
            "{summary:?}"
        );
        let proposed = proposed_claim_bindings_for_plan(&plan, &atoms, &summary.text, &[15]);
        let bindings = verified_claim_bindings_for_plan(&plan, &atoms, &summary.text, &[15]);
        let claim_terms = informative_terms(&summary.text);
        let claim_content = claim_content_terms(
            &claim_terms,
            &plan
                .subject
                .iter()
                .map(|subject| subject.to_lowercase())
                .collect::<Vec<_>>(),
        );
        let misses = claim_content
            .iter()
            .filter(|term| !text_matches_term(&summary.text.to_lowercase(), term))
            .cloned()
            .collect::<Vec<_>>();
        assert!(!bindings.0.is_empty(), "plan={plan:?}; atoms={atoms:#?}; terms={claim_terms:?}; content={claim_content:?}; misses={misses:?}; proposed={proposed:?}; bindings={bindings:?}");
    }

    #[test]
    fn final_claim_requires_durable_subject_attribution() {
        let mut atoms = project_source(ProjectionInput {
            memory_id: 8,
            content: "D8:1 Melanie: Thanks, Caroline. I built a garden.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        for atom in &mut atoms {
            atom.supports_slots = vec!["answer_bearing_fact".to_string(), "action".to_string()];
        }

        let plan = build_query_plan("What did Caroline build?", &[]);
        assert!(
            verified_claim_bindings_for_plan(&plan, &atoms, "Caroline built a garden.", &[8],)
                .0
                .is_empty()
        );
    }

    #[test]
    fn special_claim_paths_apply_the_leading_subject_guard() {
        let count_plan = build_query_plan("How many times did John inspect the generator?", &[]);
        for claim in [
            "Direct answer: Caroline inspected the generator twice.",
            "Partial answer: Caroline inspected the generator twice.",
            "Likely answer: Caroline inspected the generator twice.",
            "Inferred answer: Caroline inspected the generator twice.",
            "Direct answer: Yesterday, Caroline inspected the generator twice.",
            "Direct answer: caroline inspected the generator twice.",
            "Direct answer: John inspected the generator; Caroline inspected it twice.",
        ] {
            assert!(
                !claim_subject_matches_plan(&count_plan, claim),
                "mismatched special Claim passed subject guard: {claim}"
            );
        }

        let mut atoms = project_source(ProjectionInput {
            memory_id: 11,
            content: "D11:1 John: I inspected the generator twice.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        for atom in &mut atoms {
            if atom.kind.grounding_role() == GroundingRole::Original {
                atom.supports_slots = vec![
                    "answer_bearing_fact".to_string(),
                    "action".to_string(),
                    "object".to_string(),
                ];
            }
        }
        assert!(verified_claim_bindings_for_plan(
            &count_plan,
            &atoms,
            "Direct answer: Caroline inspected the generator twice.",
            &[11],
        )
        .0
        .is_empty());
    }

    #[test]
    fn condition_direction_and_negation_scope_require_the_same_relation() {
        let plan = build_query_plan("What did John not deploy unless the sensor failed?", &[]);
        assert_eq!(plan.condition_relation, ConditionRelation::Unless);
        assert!(strict_condition_terms_match_evidence(
            &plan,
            "john did not deploy the drone unless the sensor failed."
        ));
        assert!(!strict_condition_terms_match_evidence(
            &plan,
            "john did not deploy the drone if the sensor failed."
        ));

        let answer_value = vec!["drone".to_string()];
        assert!(negation_scope_matches_plan(
            &plan,
            "john did not deploy the drone unless the sensor failed.",
            &answer_value,
        ));
        assert!(!negation_scope_matches_plan(
            &plan,
            "john did not rest, but deployed the drone unless the sensor failed.",
            &answer_value,
        ));
        assert!(!negation_scope_matches_plan(
            &plan,
            "john did not just deploy the drone; he repaired it too.",
            &answer_value,
        ));
    }

    #[test]
    fn count_events_require_a_date_to_distinguish_undated_variants() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        let mut undated = project_source(ProjectionInput {
            memory_id: 21,
            content: "D21:1 John: I inspected the generator carefully.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        undated.extend(project_source(ProjectionInput {
            memory_id: 22,
            content: "D22:1 John: I carefully inspected the generator.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        }));
        for atom in &mut undated {
            if atom.kind.grounding_role() == GroundingRole::Original {
                atom.supports_slots = vec![
                    "answer_bearing_fact".to_string(),
                    "action".to_string(),
                    "object".to_string(),
                ];
            }
        }
        assert!(
            !count_event_set(&plan, &undated, None).is_proven(),
            "undated paraphrases must not be promoted to two events"
        );

        let mut dated = project_source(ProjectionInput {
            memory_id: 23,
            content: "Session date: 10 May 2023\nD23:1 John: I inspected the generator last week.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        dated.extend(project_source(ProjectionInput {
            memory_id: 24,
            content: "Session date: 24 May 2023\nD24:1 John: I inspected the generator last week.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        }));
        for atom in &mut dated {
            if atom.kind.grounding_role() == GroundingRole::Original {
                atom.supports_slots = vec![
                    "answer_bearing_fact".to_string(),
                    "action".to_string(),
                    "object".to_string(),
                ];
            }
        }
        // A week is an interval, not an exact day suitable for event identity.
        assert!(!count_event_set(&plan, &dated, None).is_proven());
    }

    fn proof_test_atoms(memory_id: u64, content: &str) -> Vec<EvidenceAtom> {
        let mut atoms = project_source(ProjectionInput {
            memory_id,
            content,
            source: "unit-test",
            entity: None,
            score: 0.9,
            is_event: false,
        });
        for atom in &mut atoms {
            if atom.kind.grounding_role() == GroundingRole::Original {
                atom.supports_slots = vec![
                    "answer_bearing_fact".into(),
                    "action".into(),
                    "object".into(),
                ];
            }
        }
        atoms
    }

    #[test]
    fn systematic_review_question_and_source_roles_do_not_follow_owners() {
        let records = [test_item(
            1,
            0.9,
            "John: I gave a book.\nCaroline: I received it.",
        )];
        assert_eq!(
            discover_subject("What did John give Caroline?", &records).as_deref(),
            Some("John")
        );
        let owner = fact_attribution("I inspected Caroline's generator twice.", Some("John"));
        assert_eq!(owner.fact_subjects, vec!["John"]);
        assert!(owner.possessive_owners.contains(&"Caroline".into()));
        let agent = fact_attribution("Caroline inspected my generator twice.", Some("John"));
        assert_eq!(agent.fact_subjects, vec!["Caroline"]);
    }

    #[test]
    fn copular_question_predicates_do_not_become_fact_subjects() {
        let records = [test_item(
            1,
            0.9,
            "John: I read the deployment handbook.",
        )];
        for query in [
            "Which deployment region is recorded?",
            "Which account was assigned?",
            "Which account was currently assigned?",
            "What is recorded?",
        ] {
            assert_eq!(discover_subject(query, &records), None, "{query}");
        }
        for (query, expected) in [
            ("Which book was John reading?", "John"),
            ("Which book was deployment reading?", "deployment"),
            ("Which book was Alice assigned?", "Alice"),
            ("Where is deployment?", "deployment"),
            ("What is Alice's region?", "Alice"),
        ] {
            assert_eq!(
                discover_subject(query, &records).as_deref(),
                Some(expected),
                "{query}"
            );
        }

        let plan = build_query_plan("Which deployment region is recorded?", &records);
        assert_eq!(plan.subject, None);
        assert!(!plan.object_terms.iter().any(|term| term == "recorded"));
        assert!(plan.object_terms.iter().any(|term| term == "deployment"));
        assert!(plan.object_terms.iter().any(|term| term == "region"));
        assert!(!plan.required_slots.iter().any(|slot| slot == "subject"));

        let deployment = [test_item(2, 0.95, "The deployment region is Shanghai.")];
        let pipeline = plan_answer_pipeline(
            "Which deployment region is recorded?",
            &deployment,
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        let claims = &pipeline
            .unified_draft
            .as_ref()
            .expect("scalar direct answer")
            .rendered_claims;
        assert_eq!(claims.len(), 1);
        let coverage = pipeline
            .unified_draft_coverage()
            .expect("structured scalar Claim coverage");
        assert_eq!(coverage.len(), 1);
        assert!(coverage[0].covered, "{coverage:#?}");
    }

    #[test]
    fn cjk_lexical_current_does_not_create_condition_or_quantity_slots() {
        let plan = build_query_plan("当前项目是什么", &[]);
        assert_eq!(plan.condition_relation, ConditionRelation::None);
        assert!(plan.condition_terms.is_empty());
        assert!(!plan.quantity_terms.iter().any(|term| term == "前"));
        assert!(!plan.required_slots.iter().any(|slot| slot == "condition"));

        for query in ["当然可以吗", "当地天气是什么", "当时发生了什么"] {
            assert_eq!(
                extract_condition_relation(query),
                ConditionRelation::None,
                "{query}"
            );
        }
        assert_eq!(
            extract_condition_relation("当传感器失效时，系统是否停机"),
            ConditionRelation::If
        );
        assert!(source_condition_clause("agent 当前项目是语义记忆改造").is_none());
        assert_eq!(
            source_condition_clause("当传感器失效时，系统停机")
                .map(|(relation, _)| relation),
            Some(ConditionRelation::If)
        );
        let bounded = extract_quantity_terms("列出前 3 项");
        assert!(bounded.iter().any(|term| term == "3"));
        assert!(bounded.iter().any(|term| term == "前"));
        assert!(!extract_quantity_terms("当前项目").iter().any(|term| term == "前"));
    }

    #[test]
    fn systematic_review_preserves_decimal_spans_and_quantities() {
        assert_eq!(
            split_statements("The voltage was 3.7 V. Dr. Chen inspected it."),
            vec!["The voltage was 3.7 V.", "Dr. Chen inspected it."]
        );
        for (claim, source) in [
            ("2 V", "12 V"),
            ("-2 V", "2 V"),
            ("2 A", "2 V"),
            ("3.7 V", "37 V"),
            ("2%", "2 V"),
            ("2 V", "2foo V"),
            ("2 V", "2e3 V"),
            ("2 mV", "2 MV"),
        ] {
            assert!(
                !numeric_values_supported(claim, source),
                "{claim} / {source}"
            );
        }
        assert!(numeric_values_supported("3.7 V", "The voltage was 3.7 V."));
        assert!(numeric_values_supported("3.70 volts", "+03.7 V"));
        assert!(numeric_values_supported("2%", "The rate is 2%."));
    }

    #[test]
    fn systematic_review_final_actor_and_predicate_reject_substitutions() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        let atoms = proof_test_atoms(201, "D201:1 John: I inspected the generator twice.");
        for claim in [
            "Inspected twice. By Caroline.",
            "Inspected twice. Performed by Caroline.",
            "John repaired the generator twice.",
        ] {
            assert!(
                verified_claim_bindings_for_plan(&plan, &atoms, claim, &[201])
                    .0
                    .is_empty()
            );
        }
        assert!(claim_subject_matches_plan(
            &plan,
            "John inspected twice by May 12."
        ));
        let plan = build_query_plan("What did John inspect?", &[]);
        let atoms = proof_test_atoms(202, "D202:1 John: I inspected the generator.");
        assert!(verified_claim_bindings_for_plan(
            &plan,
            &atoms,
            "John repaired the generator.",
            &[202]
        )
        .0
        .is_empty());
        assert!(!verified_claim_bindings_for_plan(
            &plan,
            &atoms,
            "John inspected the generator.",
            &[202]
        )
        .0
        .is_empty());
    }

    #[test]
    fn systematic_review_counts_include_zero_and_scoped_components() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        for (text, expected) in [
            ("十一次", 11),
            ("三十一次", 31),
            ("0 times", 0),
            ("once in Paris and twice in Rome", 3),
        ] {
            let atoms = proof_test_atoms(
                203,
                &format!("D203:1 John: I inspected the generator {text}."),
            );
            let count = count_event_set(&plan, &atoms, None);
            assert!(count.is_proven(), "{text}: {count:?}");
            assert_eq!(count.numeric_value, expected);
            let draft = build_count_unified_answer(&plan, &atoms, false, false).unwrap();
            assert!(
                !verified_claim_bindings_for_plan(&plan, &atoms, &draft.text, &[203])
                    .0
                    .is_empty(),
                "{}",
                draft.text
            );
        }
        assert!(count_quantity_value("once or twice").is_err());
        assert!(
            count_quantity_value("three times in total, once in Paris and twice in Rome").is_err()
        );
        let atoms = proof_test_atoms(
            203,
            "D203:1 John: I inspected the generator twice last week.",
        );
        assert_eq!(count_event_set(&plan, &atoms, None).numeric_value, 2);
    }

    #[test]
    fn systematic_review_counts_share_frozen_scope_and_conditions() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        let mut atoms = proof_test_atoms(
            204,
            "D204:1 John: I inspected the generator on 10 May 2023.",
        );
        atoms.extend(proof_test_atoms(
            205,
            "D205:1 John: I inspected the generator on 12 May 2023.",
        ));
        assert!(
            verified_claim_bindings_for_plan(&plan, &atoms, "Direct answer: 1.", &[204])
                .0
                .is_empty()
        );
        for marker in ["when", "as long as", "only after"] {
            let atoms = proof_test_atoms(
                206,
                &format!(
                    "D206:1 John: I inspected the generator twice {marker} the sensor passed."
                ),
            );
            let draft = build_count_unified_answer(&plan, &atoms, false, false).unwrap();
            assert!(
                !verified_claim_bindings_for_plan(&plan, &atoms, &draft.text, &[206])
                    .0
                    .is_empty(),
                "{}",
                draft.text
            );
            assert!(verified_claim_bindings_for_plan(
                &plan,
                &atoms,
                &draft.text.replace("passed", "failed"),
                &[206]
            )
            .0
            .is_empty());
        }
    }

    #[test]
    fn duration_proof_preserves_signs_dates_and_conflicts() {
        let query = "How many days passed between the launch and the review?";
        let items = vec![
            test_item(
                401,
                1.0,
                "Session date: 2025-03-30\nD1:1 Mira: The launch happened on March 1st.",
            ),
            test_item(
                402,
                1.0,
                "Session date: 2025-03-30\nD2:1 Mira: The review happened on March 10th.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, Some(1), "concise", 2, false);
        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("bounded date calculation");
        assert!(
            draft.text.contains("Elapsed time: 9 days"),
            "{}",
            draft.text
        );
        assert!(pipeline
            .verify_claim_evidence("Elapsed time: 9 days.", &HashSet::from([401, 402]))
            .is_bound());
        for claim in [
            "Elapsed time: -9 days.",
            "Elapsed time: 10 days.",
            "First event: the launch on 20250-3-01.",
        ] {
            assert!(
                !pipeline
                    .verify_claim_evidence(claim, &HashSet::from([401, 402]))
                    .is_bound(),
                "{claim}"
            );
        }
        assert!(!pipeline
            .verify_claim_evidence("Elapsed time: 9 days.", &HashSet::from([401]))
            .is_bound());
        let mut conflicting = items.clone();
        conflicting.push(test_item(
            403,
            0.1,
            "D3:1 Mira: The launch happened on March 2nd, 2025.",
        ));
        let conflict =
            plan_answer_pipeline(query, &conflicting, 3, 3, Some(1), "concise", 2, false);
        assert!(!conflict
            .verify_claim_evidence("Elapsed time: 9 days.", &HashSet::from([401, 402]))
            .is_bound());
    }

    #[test]
    fn registered_operators_declare_structured_contracts() {
        let plan = build_query_plan("What happened?", &[]);
        let operators = build_operator_plan(&plan, true);
        let mut contracts = BTreeMap::new();
        for operator in operators {
            contracts.insert(operator.as_str(), operator.contract());
        }
        assert_eq!(contracts.len(), 29);
        assert_eq!(
            contracts
                .values()
                .filter(|contract| **contract == AnswerOperatorContract::CandidateClaims)
                .count(),
            26
        );
        assert_eq!(
            contracts["unified_answer_generator_v1"],
            AnswerOperatorContract::CheckedPresentation
        );
        assert_eq!(
            contracts["novel_chapter_unavailable_v1"],
            AnswerOperatorContract::CoverageFeedback
        );
        assert_eq!(
            contracts["answerability_gate_v1"],
            AnswerOperatorContract::CoverageFeedback
        );
    }

    #[test]
    fn structural_ranges_keep_middle_chapters_without_expansion_or_false_completeness() {
        let query = "比较所有小说第1章到第3章讲了什么。";
        let selectors = extract_structural_selectors(query);
        assert!(selectors.contains(&"chapter:1..3".into()));
        assert!(matching_structural_selectors("Section: 第2章", &selectors).contains(&"chapter:1..3".into()));
        assert!(matching_structural_selectors("Section: 第4章", &selectors).is_empty());
        assert!(matching_structural_selectors("Section: 第2节", &selectors).is_empty());
        assert!(extract_structural_selectors("第1章到第18446744073709551615章").len() <= 3);
        let source = test_item(601, 1.0, "Novel: A\nSection: 第2章\n林岚进入旧城。");
        let initial = plan_answer_pipeline(query, &[source.clone()], 1, 1, Some(1), "auto", 4, true);
        assert!(initial.verify_claim_evidence("林岚进入旧城。", &HashSet::from([601])).is_bound(),
            "initial plan {:?}; retained atoms {:?}; rejected {:?}", initial.query_plan,
            initial.evidence_atoms, initial.receipt.evidence_plan.rejected_atoms);
        let mut pipeline = plan_answer_pipeline(query, &[], 0, 0, Some(1), "auto", 4, true);
        pipeline.admit_renderer_evidence(&[source]);
        assert!(pipeline.verify_claim_evidence("林岚进入旧城。", &HashSet::from([601])).is_bound());
        let atoms = pipeline.evidence_atoms.iter().collect::<Vec<_>>();
        assert!(!structural_range_fully_observed("chapter:1..3", &atoms));
        assert!(!structural_range_fully_observed("chapter:0..18446744073709551615", &atoms));
        assert!(pipeline.receipt.slot_coverage.iter().any(|slot|
            slot.slot == "structural_selector:chapter:1..3" && slot.status != "covered"));
        let first = test_item(602, 1.0, "Section: 第1章\n林岚抵达灯塔。");
        let last = test_item(603, 1.0, "Section: 第3章\n林岚返回港口。");
        pipeline.admit_renderer_evidence(&[first.clone(), last]);
        assert!(pipeline.receipt.slot_coverage.iter().any(|slot|
            slot.slot == "structural_selector:chapter:1..3" && slot.status == "covered"));
        let mut conflict = first;
        conflict.version += 1;
        pipeline.admit_renderer_evidence(&[conflict]);
        assert!(pipeline.receipt.slot_coverage.iter().any(|slot|
            slot.slot == "structural_selector:chapter:1..3" && slot.status == "partial"));
        assert!(pipeline.receipt.slot_coverage.iter().any(|slot|
            slot.slot == "source_snapshot" && slot.status == "partial"));
    }

    #[test]
    fn structural_comparison_keeps_document_facts_and_content_constraints() {
        for query in ["比较所有小说的第1章。", "请对比全部文件的第十二节。",
            "比较所有小说第1章到第2章讲了什么。", "对比全部文档第一节至第三节发生了什么？",
            "比较各个作品第1章和第2章。"]
        {
            assert!(question_informative_terms(query).is_empty(), "{query}");
        }
        for query in [
            "比较所有小说的第1章中林岚的任务。",
            "比较所有小说中林岚的第1章。",
            "比较所有小说第1章到第2章中林岚的任务。",
            "比较所有小说第1章中林岚的任务和第2章。",
        ] {
            assert!(
                question_informative_terms(query).contains(&"林岚".to_string()),
                "{query}"
            );
        }
        let sources = vec![
            test_item(601, 1.0, "Novel: A\nSection: 第1章\n林岚进入旧城。"),
            test_item(602, 1.0, "Novel: B\nSection: 第1章\n林岚抵达灯塔。"),
            test_item(603, 1.0, "Novel: C\nSection: 第2章\n林岚抵达港口。"),
        ];
        let mut pipeline =
            plan_answer_pipeline("比较所有小说的第1章。", &[], 0, 0, None, "auto", 4, true);
        pipeline.admit_renderer_evidence(&sources);
        assert!(pipeline
            .verify_claim_evidence("林岚进入旧城。", &HashSet::from([601]))
            .is_bound());
        assert!(pipeline
            .verify_claim_evidence("林岚抵达灯塔。", &HashSet::from([602]))
            .is_bound());
        assert!(!pipeline
            .verify_claim_evidence("林岚抵达港口。", &HashSet::from([603]))
            .is_bound());
        assert!(!pipeline
            .verify_claim_evidence("林岚抵达灯塔。", &HashSet::from([601]))
            .is_bound());
    }

    #[test]
    fn structural_question_format_masking_preserves_content_constraints() {
        assert!(
            question_informative_terms("第一章发生了什么？请给出章节概要、关键事件和证据。")
                .is_empty()
        );
        assert!(question_informative_terms("第十二节发生了什么？提供摘要与引用。").is_empty());
        let content = question_informative_terms("第一章发生了什么？请给出林岚拒绝任务的证据。");
        assert!(content.contains(&"林岚".to_string()));
        assert!(content.contains(&"拒绝".to_string()));
        assert!(content.contains(&"任务".to_string()));
        let source = test_item(502, 1.0, "Section: 第2章\n林岚接下清扫任务。");
        let mut pipeline = plan_answer_pipeline(
            "第一章发生了什么？请给出概要和证据。",
            &[],
            0,
            0,
            None,
            "auto",
            4,
            true,
        );
        pipeline.admit_renderer_evidence(&[source]);
        assert!(!pipeline
            .verify_claim_evidence("林岚接下清扫任务。", &HashSet::from([502]))
            .is_bound());
    }

    #[test]
    fn chapter_claim_admission_binds_original_chapter_fields() {
        let source = test_item(501, 1.0, "memory_type: novel_chapter_summary\nNovel: 雾都档案\nSection: 第1章 清扫日\nChapter summary: 林岚进入雾都清扫区，发现回收任务背后有异常污染线索。\nKey events:\n- 林岚接下清扫任务\n- 清扫区出现异常污染线索\nEvidence:\n- 回收站记录把任务标成高风险\nSource chunks: 1");
        let mut pipeline = plan_answer_pipeline(
            "第一章发生了什么？请给出章节概要、关键事件和证据。",
            &[],
            0,
            0,
            None,
            "auto",
            4,
            true,
        );
        pipeline.admit_renderer_evidence(&[source]);
        let claim = "林岚接下清扫任务";
        let proof = pipeline.verify_claim_evidence(claim, &HashSet::from([501]));
        let atoms = pipeline
            .evidence_atoms
            .iter()
            .map(|atom| {
                (
                    &atom.text,
                    &atom.supports_slots,
                    &atom.section_scope,
                    &atom.fact_subjects,
                )
            })
            .collect::<Vec<_>>();
        assert!(
            proof.is_bound(),
            "plan={:?}; atoms={atoms:?}",
            pipeline.query_plan
        );
    }

    #[test]
    fn source_admission_preserves_explicit_section_scope() {
        let source = test_item(405, 0.8, "Section: 7\nThe sensor stopped recording.\nSee section 8 for details.\nSection: 8\nThe motor started.");
        let projected = project_source(ProjectionInput {
            memory_id: source.id,
            content: &source.content,
            source: &source.source,
            entity: None,
            score: 1.0,
            is_event: false,
        });
        let sensor = projected
            .iter()
            .find(|atom| atom.text.contains("sensor stopped"))
            .unwrap();
        let motor = projected
            .iter()
            .find(|atom| atom.text.contains("motor started"))
            .unwrap();
        assert_eq!(sensor.section_scope, vec!["section:7"]);
        assert_eq!(motor.section_scope, vec!["section:8"]);
        let mut pipeline = plan_answer_pipeline(
            "What happened in section 7?",
            &[],
            0,
            0,
            None,
            "auto",
            1,
            false,
        );
        pipeline.admit_renderer_evidence(&[source.clone()]);
        assert!(
            pipeline
                .verify_claim_evidence("The sensor stopped recording.", &HashSet::from([405]))
                .is_bound(),
            "{:#?}",
            pipeline.receipt
        );
        assert!(!pipeline
            .verify_claim_evidence("The motor started.", &HashSet::from([405]))
            .is_bound());
        let mut next_section = plan_answer_pipeline(
            "What happened in section 8?",
            &[],
            0,
            0,
            None,
            "auto",
            1,
            false,
        );
        next_section.admit_renderer_evidence(&[source]);
        assert!(!next_section
            .verify_claim_evidence("See section 8 for details.", &HashSet::from([405]))
            .is_bound());
        assert!(next_section
            .verify_claim_evidence("The motor started.", &HashSet::from([405]))
            .is_bound());
        let mut count = plan_answer_pipeline(
            "How many times did Mira inspect the motor?",
            &[],
            0,
            0,
            None,
            "auto",
            1,
            false,
        );
        count.admit_renderer_evidence(&[test_item(406, 1.0, "Mira: I inspected the motor twice.")]);
        assert!(count.evidence_atoms.is_empty());
    }

    #[test]
    fn renderer_admission_recovers_atoms_without_promoting_unrelated_facts() {
        let source = test_item(407, 0.8,
            "D4:1 Mira: I inspected the rotor.\nD4:2 Mira: I inspected the valve.\nD4:3 John: I inspected the pump.\nD4:4 Mira: I painted the shed.");
        let mut pipeline = plan_answer_pipeline(
            "What did Mira inspect?",
            &[source.clone()],
            1,
            1,
            None,
            "auto",
            1,
            false,
        );
        // Model an earlier selection budget that retained only the first atom.
        pipeline
            .evidence_atoms
            .retain(|atom| atom.text.contains("rotor"));
        assert_eq!(pipeline.evidence_atoms.len(), 1);
        let ids = HashSet::from([407]);
        assert!(!pipeline
            .verify_claim_evidence("Mira inspected the valve.", &ids)
            .is_bound());
        pipeline.admit_renderer_evidence(&[source.clone(), source.clone()]);
        let proof = pipeline.verify_claim_evidence("Mira inspected the valve.", &ids);
        assert!(proof.is_bound(), "{:#?}", pipeline.receipt);
        assert_eq!(proof.atom_ids, vec!["memory:407:atom:1"]);
        assert_eq!(proof.source_anchors[0].value, "D4:2");
        assert!(!pipeline
            .verify_claim_evidence("Mira inspected the pump.", &ids)
            .is_bound());
        assert!(!pipeline
            .verify_claim_evidence("Mira inspected the shed.", &ids)
            .is_bound());
        let atom_count = pipeline.evidence_atoms.len();
        let stage_count = pipeline.receipt.stage_trace.len();
        pipeline.admit_renderer_evidence(&[source]);
        assert_eq!(pipeline.evidence_atoms.len(), atom_count);
        assert_eq!(pipeline.receipt.stage_trace.len(), stage_count);
        assert_eq!(
            pipeline
                .evidence_atoms
                .iter()
                .map(|atom| &atom.atom_id)
                .collect::<HashSet<_>>()
                .len(),
            atom_count
        );
    }

    #[test]
    fn renderer_admission_rejects_snapshot_changes_and_cannot_resurrect_them() {
        let source = test_item(408, 0.8, "D4:1 Mira: I inspected the rotor.");
        for variant in 0..4 {
            let mut changed = source.clone();
            match variant {
                0 => changed.content = "D4:1 Mira: I inspected the valve.".into(),
                1 => changed.source = "different-source".into(),
                2 => changed.entity = Some("different-owner".into()),
                _ => changed.version += 1,
            }
            let mut pipeline = plan_answer_pipeline(
                "What did Mira inspect?",
                &[source.clone()],
                1,
                1,
                None,
                "auto",
                1,
                false,
            );
            assert!(pipeline
                .verify_claim_evidence("Mira inspected the rotor.", &HashSet::from([408]))
                .is_bound());
            pipeline.admit_renderer_evidence(&[changed]);
            assert!(pipeline.evidence_atoms.is_empty());
            assert!(!pipeline
                .verify_claim_evidence("Mira inspected the rotor.", &HashSet::from([408]))
                .is_bound());
            assert!(pipeline
                .receipt
                .slot_coverage
                .iter()
                .any(|slot| slot.slot == "source_snapshot" && slot.status == "partial"));
            assert!(pipeline
                .unified_draft_coverage()
                .is_none_or(|coverage| coverage.iter().all(|claim| !claim.covered)));
            pipeline.admit_renderer_evidence(&[source.clone()]);
            assert!(pipeline.evidence_atoms.is_empty());
        }
    }

    #[test]
    fn renderer_admission_rejects_conflicting_new_source_versions_in_one_batch() {
        let original = test_item(409, 0.8, "Mira: I inspected the rotor.");
        let changed = test_item(409, 0.8, "Mira: I inspected the valve.");
        for items in [
            vec![original.clone(), changed.clone()],
            vec![changed, original],
        ] {
            let mut pipeline =
                plan_answer_pipeline("What did Mira inspect?", &[], 0, 0, None, "auto", 1, false);
            pipeline.admit_renderer_evidence(&items);
            assert!(pipeline.evidence_atoms.is_empty());
            assert_eq!(
                pipeline.receipt.stage_trace.last().unwrap().status,
                "failed"
            );
        }
    }

    #[test]
    fn renderer_admission_keeps_count_scope_frozen_and_detects_revisions() {
        let source = test_item(410, 0.8, "Mira: I inspected the rotor twice.");
        let mut pipeline = plan_answer_pipeline(
            "How many times did Mira inspect the rotor?",
            &[source.clone()],
            1,
            1,
            None,
            "auto",
            1,
            false,
        );
        let count = pipeline.evidence_atoms.len();
        pipeline.admit_renderer_evidence(&[
            source.clone(),
            test_item(411, 1.0, "Mira: I inspected the rotor once."),
        ]);
        assert_eq!(pipeline.evidence_atoms.len(), count);
        assert!(pipeline
            .verify_claim_evidence("Direct answer: 2.", &HashSet::from([410]))
            .is_bound());
        let mut changed = source;
        changed.version += 1;
        pipeline.admit_renderer_evidence(&[changed]);
        assert!(!pipeline
            .verify_claim_evidence("Direct answer: 2.", &HashSet::from([410]))
            .is_bound());
    }

    #[test]
    fn renderer_admission_budget_prioritizes_relevant_atoms_and_reports_truncation() {
        let filler = test_item(412, 0.1, "The weather was pleasant.");
        let mut pipeline =
            plan_answer_pipeline("What did Mira inspect?", &[], 0, 0, None, "auto", 1, false);
        let atom = project_source(ProjectionInput {
            memory_id: filler.id,
            content: &filler.content,
            source: &filler.source,
            entity: None,
            score: filler.score,
            is_event: false,
        })
        .remove(0);
        for index in 0..MAX_RUNTIME_ATOMS - 1 {
            let mut padding = atom.clone();
            padding.atom_id = format!("padding:{index}");
            pipeline.evidence_atoms.push(padding);
        }
        pipeline.admit_renderer_evidence(&[test_item(
            413,
            0.9,
            "The weather was pleasant.\nMira: I inspected the rotor.",
        )]);
        assert_eq!(pipeline.evidence_atoms.len(), MAX_RUNTIME_ATOMS);
        assert!(pipeline
            .verify_claim_evidence("Mira inspected the rotor.", &HashSet::from([413]))
            .is_bound());
        assert!(pipeline
            .receipt
            .slot_coverage
            .iter()
            .any(|slot| slot.slot == "proof_scope" && slot.status == "partial"));
        assert_eq!(pipeline.receipt.stage_trace.last().unwrap().output_count, 1);
    }

    #[test]
    fn systematic_review_empty_coverage_clears_stale_success() {
        let pipeline = plan_answer_pipeline(
            "What did John inspect?",
            &[test_item(
                207,
                0.9,
                "D207:1 John: I inspected the generator.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        let receipt = finalize_answer_pipeline_receipt(
            pipeline.receipt,
            &[],
            "synthesized",
            "unified_answer_generator_v1",
            false,
            &[],
            Some("unified_answer_generator_v1"),
        );
        assert!(receipt.claim_plan.is_empty());
        assert!(receipt.final_evidence_ids.is_empty());
        assert_eq!(
            receipt
                .stage_trace
                .iter()
                .find(|stage| stage.stage == "answer")
                .unwrap()
                .status,
            "failed"
        );
    }

    #[test]
    fn final_proof_checks_postposed_agents_before_count_return() {
        for subject in ["John", "Mira", "Holly"] {
            let plan = build_query_plan(
                &format!("How many times did {subject} inspect the generator?"),
                &[],
            );
            let atoms = proof_test_atoms(
                120,
                &format!("D120:1 {subject}: I inspected the generator twice."),
            );
            for claim in [
                format!("Inspected twice by {subject}."),
                format!("Inspected the generator twice by {subject}."),
            ] {
                assert!(
                    !verified_claim_bindings_for_plan(&plan, &atoms, &claim, &[120])
                        .0
                        .is_empty(),
                    "{claim}"
                );
            }
            for claim in [
                "Inspected twice by Caroline.".to_string(),
                "Inspected twice by caroline.".to_string(),
                format!("{subject} inspected twice by Caroline."),
                format!("Inspected twice by {subject} and Caroline."),
            ] {
                assert!(
                    verified_claim_bindings_for_plan(&plan, &atoms, &claim, &[120])
                        .0
                        .is_empty(),
                    "{claim}"
                );
            }
        }
    }

    #[test]
    fn final_proof_checks_plain_when_instead_of_skipping_conditions() {
        let plan = build_query_plan(
            "How many times did John inspect the generator when the sensor passed?",
            &[],
        );
        // Query planning already stores embedded when-clauses as qualifiers;
        // do not reclassify time/event context as a hypothetical if-condition.
        assert_eq!(plan.condition_relation, ConditionRelation::None);
        assert_eq!(
            build_query_plan("When did John inspect the generator?", &[]).condition_relation,
            ConditionRelation::None
        );
        let atoms = proof_test_atoms(
            121,
            "D121:1 John: I inspected the generator twice when the sensor passed.",
        );
        let good = "John inspected the generator twice when the sensor passed.";
        assert!(
            !verified_claim_bindings_for_plan(&plan, &atoms, good, &[121])
                .0
                .is_empty()
        );
        for claim in [
            "John inspected the generator twice when the sensor failed.",
            "John inspected the generator twice when the sensor did not pass.",
            "John inspected the generator twice if the sensor passed.",
            "John inspected the generator twice unless the sensor passed.",
            "John inspected the generator twice.",
        ] {
            assert!(
                verified_claim_bindings_for_plan(&plan, &atoms, claim, &[121])
                    .0
                    .is_empty(),
                "{claim}"
            );
        }
        assert_eq!(
            source_condition_clause("john inspected only when the sensor passed.")
                .unwrap()
                .0,
            ConditionRelation::OnlyWhen
        );
        assert!(source_condition_clause("somewhere whenever wheneverland").is_none());
        assert!(!claim_logic_matches_source(
            &plan,
            "when the sensor failed, john inspected the generator twice.",
            "when the sensor passed, john inspected the generator twice.",
        ));
    }

    #[test]
    fn final_proof_ambiguous_event_count_cannot_fall_back_to_one() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        for text in [
            "D122:1 John: I inspected the generator on 10 May 2023 and I inspected the generator on 12 May 2023.",
            "D122:1 John: I inspected the generator on 10 May 2023, and I inspected the generator on 12 May 2023.",
            "D122:1 John: I inspected the generator on 10 May 2023 and on 12 May 2023.",
            "D122:1 John: I inspected the generator and I inspected the generator again.",
        ] {
            let mut atoms = proof_test_atoms(122, text);
            let eligible = atoms.iter().find(|atom| count_atom_matches_plan(&plan, atom)).expect("attributed original event");
            assert_eq!(count_event_date_resolution(&plan, eligible, &HashMap::new()), CountEventDate::Ambiguous, "{text}");
            assert!(count_event_identity(&plan, eligible, &HashMap::new()).is_empty());
            let count = count_event_set(&plan, &atoms, None);
            assert!(!count.is_proven(), "{count:#?}");
            assert!(count.derivation.iter().any(|reason| reason == "count_event_cardinality_unresolved"));
            assert!(build_count_unified_answer(&plan, &atoms, false, false).is_none());
            assert!(verified_claim_bindings_for_plan(&plan, &atoms, "Direct answer: 1.", &[122]).0.is_empty());
            // A known occurrence must not mask an unresolved contributor.
            atoms.extend(proof_test_atoms(123, "D123:1 John: I inspected the generator on 15 May 2023."));
            assert!(!count_event_set(&plan, &atoms, None).is_proven());
            assert!(verified_claim_bindings_for_plan(&plan, &atoms, "Direct answer: 2.", &[122,123]).0.is_empty());
        }
        let single = proof_test_atoms(124, "D124:1 John: I inspected the generator.");
        let event = single
            .iter()
            .find(|atom| count_atom_matches_plan(&plan, atom))
            .unwrap();
        assert_eq!(
            count_event_date_resolution(&plan, event, &HashMap::new()),
            CountEventDate::Missing
        );
        assert_eq!(count_event_set(&plan, &single, None).numeric_value, 1);
        let mut separate = proof_test_atoms(
            125,
            "D125:1 John: I inspected the generator on 10 May 2023.",
        );
        separate.extend(proof_test_atoms(
            126,
            "D126:1 John: I inspected the generator on 12 May 2023.",
        ));
        assert_eq!(count_event_set(&plan, &separate, None).numeric_value, 2);
        assert!(!verified_claim_bindings_for_plan(
            &plan,
            &separate,
            "Direct answer: 2.",
            &[125, 126]
        )
        .0
        .is_empty());
    }

    #[test]
    fn final_proof_accepts_the_renderers_chinese_and_english_count_scalars() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        for marker in [
            "once",
            "twice",
            "one time",
            "two times",
            "three times",
            "5 times",
            "一次",
            "两次",
            "二次",
            "三次",
            "四次",
            "五次",
            "12次",
        ] {
            let atoms = proof_test_atoms(
                127,
                &format!("D127:1 John: I inspected the generator {marker}."),
            );
            for chinese in [false, true] {
                let draft =
                    build_count_unified_answer(&plan, &atoms, false, chinese).expect("count draft");
                let claim = draft.rendered_claims[0].text.as_deref().unwrap();
                assert!(
                    !verified_claim_bindings_for_plan(&plan, &atoms, claim, &[127])
                        .0
                        .is_empty(),
                    "{claim}"
                );
            }
        }
        let atoms = proof_test_atoms(127, "D127:1 John: I inspected the generator 两次.");
        for claim in [
            "直接回答：三次。",
            "直接回答：Caroline 两次。",
            "直接回答：两次和三次。",
            "Caroline's total: 两次",
        ] {
            assert!(
                verified_claim_bindings_for_plan(&plan, &atoms, claim, &[127])
                    .0
                    .is_empty(),
                "{claim}"
            );
        }
    }

    #[test]
    fn final_proof_nominal_counts_bind_owner_and_keep_predicate_modifiers() {
        for subject in ["John", "Mira", "Holly"] {
            let plan = build_query_plan(
                &format!("How many times did {subject} inspect the generator?"),
                &[],
            );
            let source = format!("D110:1 {subject}: I inspected the generator twice.");
            let atoms = proof_test_atoms(110, &source);
            for claim in [
                "2.".to_string(),
                format!("{subject}'s total: 2"),
                format!("{subject}\u{2019}s count: 2"),
                format!("{subject} carefully inspected the generator twice."),
                format!("Yesterday, {subject} very carefully inspected the generator twice."),
            ] {
                assert!(
                    !verified_claim_bindings_for_plan(&plan, &atoms, &claim, &[110])
                        .0
                        .is_empty(),
                    "valid Claim: {claim}"
                );
            }
            for claim in [
                "Caroline's total: 2",
                "caroline's count: 2",
                "Caroline: 2",
                "Caroline carefully inspected the generator twice.",
                "John's total: 2; Caroline's total: 2",
            ] {
                assert!(
                    verified_claim_bindings_for_plan(&plan, &atoms, claim, &[110])
                        .0
                        .is_empty(),
                    "wrong owner: {claim}"
                );
            }
        }
    }

    #[test]
    fn final_proof_condition_order_preserves_subject_object_and_connectives() {
        let plan = build_query_plan("What did John deploy?", &[]);
        for (claim, source) in [
            (
                "john deployed the drone if the motor disabled the sensor.",
                "i deployed the drone if the sensor disabled the motor.",
            ),
            (
                "john deployed the drone if alpha starts and beta stops.",
                "i deployed the drone if beta starts and alpha stops.",
            ),
            (
                "john deployed the drone if the sensor failed or the motor stopped.",
                "i deployed the drone if the sensor failed and the motor stopped.",
            ),
            ("如果电机关闭传感器", "如果传感器关闭电机"),
        ] {
            assert!(
                !claim_logic_matches_source(&plan, claim, source),
                "role/logic reversal: {claim}"
            );
        }
        assert!(claim_logic_matches_source(
            &plan,
            "john deployed the drone if the sensor disabled the motor.",
            "i deployed the drone if the sensor disabled the motor."
        ));
    }

    #[test]
    fn final_proof_event_date_belongs_to_action_not_first_calendar_mention() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 12);
        for text in [
            "D111:1 John: After the repair on 10 May 2023, I inspected the generator on 12 May 2023.",
            "D111:1 John: I inspected the generator on 12 May 2023 after the repair on 10 May 2023.",
            "D111:1 John: After the repair on May 10, 2023, I inspected the generator on May 12, 2023.",
        ] {
            let mut atoms = proof_test_atoms(111, text);
            let action = atoms.iter().find(|atom| atom.text.contains("inspected")).unwrap();
            assert_eq!(count_event_date(&plan, action, &HashMap::new()), expected, "{text}");
            atoms.extend(proof_test_atoms(112, "D112:1 John: I inspected the generator on 12 May 2023."));
            let count = count_event_set(&plan, &atoms, None);
            assert!(!count.is_proven(), "same-day identity is still ambiguous, not two proven events: {count:#?}");
        }
        let atoms = proof_test_atoms(
            113,
            "D113:1 John: I inspected the generator between 10 May 2023 and 12 May 2023.",
        );
        assert_eq!(count_event_date(&plan, &atoms[0], &HashMap::new()), None);
    }

    #[test]
    fn final_proof_concurrent_activity_retains_its_date_without_guessing() {
        let plan = build_query_plan("How many times did John injure his ankle?", &[]);
        let expected = chrono::NaiveDate::from_ymd_opt(2023, 5, 12);
        for text in [
            "D114:1 John: I hurt my ankle while skiing on 12 May 2023.",
            "D114:1 John: I injured my ankle during training on 12 May 2023.",
        ] {
            let atoms = proof_test_atoms(114, text);
            assert_eq!(
                count_event_date(&plan, &atoms[0], &HashMap::new()),
                expected
            );
        }
        let atoms = proof_test_atoms(
            114,
            "D114:1 John: I hurt my ankle on 12 May 2023 while recalling skiing on 10 May 2023.",
        );
        assert_eq!(count_event_date(&plan, &atoms[0], &HashMap::new()), None);
    }

    #[test]
    fn final_proof_checks_claim_conditions_and_polarity_in_both_directions() {
        let mut plan = build_query_plan("What did John deploy?", &[]);
        plan.action_terms = vec!["deploy".into()];
        assert!(claim_logic_matches_source(
            &plan,
            "john deployed the drone.",
            "i deployed the drone."
        ));
        for (claim, source) in [
            ("john deployed the drone.", "i did not deploy the drone."),
            ("john did not deploy the drone.", "i deployed the drone."),
            (
                "john did not deploy the drone unless the sensor failed.",
                "i did not deploy the drone if the sensor failed.",
            ),
            (
                "john deployed the drone if the sensor failed.",
                "i deployed the drone if the sensor did not fail.",
            ),
        ] {
            assert!(
                !claim_logic_matches_source(&plan, claim, source),
                "{claim} / {source}"
            );
        }
        let atoms = proof_test_atoms(90, "D90:1 John: I did not deploy the drone.");
        assert!(
            verified_claim_bindings_for_plan(&plan, &atoms, "John deployed the drone.", &[90])
                .0
                .is_empty()
        );
    }

    #[test]
    fn final_proof_checks_count_branch_and_nonleading_actor() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        let atoms = proof_test_atoms(91, "D91:1 John: I inspected the generator twice.");
        assert!(!verified_claim_bindings_for_plan(
            &plan,
            &atoms,
            "John inspected the generator twice.",
            &[91]
        )
        .0
        .is_empty());
        for claim in [
            "Yesterday, Caroline inspected the generator twice.",
            "caroline inspected the generator twice.",
        ] {
            assert!(
                verified_claim_bindings_for_plan(&plan, &atoms, claim, &[91])
                    .0
                    .is_empty()
            );
        }
    }

    #[test]
    fn final_proof_same_day_events_keep_order_or_remain_unproven() {
        let plan = build_query_plan("How many times did John inspect the generator?", &[]);
        let mut atoms = proof_test_atoms(
            92,
            "D92:1 John: I inspected the generator on 10 May 2023 before training.",
        );
        atoms.extend(proof_test_atoms(
            93,
            "D93:1 John: I inspected the generator on 10 May 2023 after training.",
        ));
        let events = count_event_set(&plan, &atoms, None);
        assert!(events.is_proven(), "{events:#?}");
        assert_eq!(events.numeric_value, 2);
        let mut ambiguous = proof_test_atoms(
            94,
            "D94:1 John: I inspected the generator carefully on 10 May 2023.",
        );
        ambiguous.extend(proof_test_atoms(
            95,
            "D95:1 John: I quickly inspected the generator on 10 May 2023.",
        ));
        assert!(!count_event_set(&plan, &ambiguous, None).is_proven());
    }

    #[test]
    fn final_proof_activity_labels_alone_do_not_distinguish_occurrences() {
        let plan = build_query_plan("How many times did John injure his ankle?", &[]);
        let mut atoms = proof_test_atoms(98, "D98:1 John: I hurt my ankle while skiing.");
        atoms.extend(proof_test_atoms(
            99,
            "D99:1 John: I injured my ankle during training.",
        ));
        assert!(!count_event_set(&plan, &atoms, None).is_proven());
    }

    #[test]
    fn final_proof_shared_entity_or_date_is_not_an_event_link() {
        let plan = build_query_plan("Where did John inspect the cargo drone?", &[]);
        let left = proof_test_atoms(
            96,
            "D96:1 John: I inspected the cargo drone in Paris on 10 May 2023.",
        );
        for text in [
            "D97:1 John: I repaired the cargo drone in Rome on 11 May 2023.",
            "D97:1 John: I repaired the cargo drone in Rome on 10 May 2023.",
        ] {
            let right = proof_test_atoms(97, text);
            assert!(evidence_chain_relation(&plan, &left[0], &right[0]).is_empty());
        }
    }

    #[test]
    fn evidence_chain_rejects_scattered_same_subject_terms_without_a_shared_relation() {
        let plan = build_query_plan("Where did Caroline move from?", &[]);
        let atoms = project_source(ProjectionInput {
            memory_id: 25,
            content: "D25:1 Caroline: I moved from my old country after college.\nD25:2 Caroline: My country house is quiet and old.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        assert!(evidence_chain_relation(&plan, &atoms[0], &atoms[1]).is_empty());
    }

    #[test]
    fn nearby_same_subject_atoms_do_not_form_a_chain_without_two_relational_terms() {
        let plan = build_query_plan("Where is Caroline from?", &[]);
        let atoms = project_source(ProjectionInput {
            memory_id: 9,
            content: "D9:1 Caroline: I had an ankle injury.\nD9:2 Caroline: My injury needs physical therapy.",
            source: "unit-test",
            entity: None,
            score: 0.8,
            is_event: false,
        });
        assert!(evidence_chain_relation(&plan, &atoms[0], &atoms[1]).is_empty());
    }

    #[test]
    fn unified_coverage_cannot_verify_a_record_without_a_bound_atom() {
        let mut pipeline = plan_answer_pipeline(
            "What did John build?",
            &[test_item(10, 0.9, "D10:1 John: I built a greenhouse.")],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        pipeline.unified_draft = Some(UnifiedAnswerDraft {
            text: "Direct answer: John built a shed.".to_string(),
            draft_mode: "synthesized",
            rendered_claims: vec![AnswerClaimPlanReceipt {
                claim_id: "claim-1".to_string(),
                claim_type: "fact".to_string(),
                text: Some("John built a shed.".to_string()),
                required_slots: pipeline.receipt.query_plan.required_slots.clone(),
                evidence_ids: vec![10],
                evidence_atom_ids: vec!["memory:10:atom:0".to_string()],
                evidence_source_anchors: Vec::new(),
                derivation: Vec::new(),
                support_status: "planned".to_string(),
            }],
        });
        let coverage = pipeline.unified_draft_coverage().unwrap();
        assert!(!coverage[0].covered, "{coverage:#?}");
        assert!(coverage[0].evidence_ids.is_empty(), "{coverage:#?}");
        assert!(coverage[0].evidence_atom_ids.is_empty(), "{coverage:#?}");
    }

    #[test]
    fn question_plan_preserves_month_year_and_person_boundaries() {
        let people_items = [
            test_item(1, 0.8, "D1:1 Tim: I played basketball yesterday."),
            test_item(2, 0.8, "D1:2 John: My ankle is recovering."),
        ];
        let count_plan =
            build_query_plan("How many times has John injured his ankle?", &people_items);
        assert_eq!(count_plan.people, vec!["John"]);
        assert!(!count_plan
            .required_slots
            .contains(&"person:tim".to_string()));
        assert_eq!(count_plan.answer_shape, AnswerShape::CountAggregate);

        let temporal_plan = build_query_plan(
            "What new activity has Tim started learning in August 2023?",
            &people_items,
        );
        assert_eq!(temporal_plan.temporal_scope, "bounded");
        assert!(temporal_plan.time_terms.contains(&"august".to_string()));
        assert!(temporal_plan.time_terms.contains(&"2023".to_string()));
        assert!(temporal_plan
            .required_slots
            .contains(&"temporal_scope".to_string()));

        let elapsed_plan = build_query_plan(
            "How many days had passed between the Sunday mass and the Ash Wednesday service?",
            &people_items,
        );
        assert_eq!(elapsed_plan.answer_target, AnswerTarget::Duration);
        assert_eq!(elapsed_plan.answer_shape, AnswerShape::CountAggregate);
    }

    #[test]
    fn question_plan_tracks_requested_time_precision_and_list_entry_class() {
        let items = [test_item(1, 0.8, "D1:1 John: I like fantasy films.")];

        let temporal = build_query_plan("Which month was John in Italy?", &items);
        assert_eq!(temporal.temporal_granularity, TemporalGranularity::Month);
        assert!(temporal.required_slots.contains(&"event_time".to_string()));

        let list = build_query_plan("What fantasy movies does John like?", &items);
        assert_eq!(list.answer_class_terms, vec!["movies"]);
        assert!(list.required_slots.contains(&"answer_class".to_string()));

        let generic = build_query_plan("What aspects does John like?", &items);
        assert!(generic.answer_class_terms.is_empty());
    }

    #[test]
    fn exact_relative_event_dates_keep_the_event_day_separate_from_the_anchor() {
        let session_date = chrono::NaiveDate::from_ymd_opt(2023, 11, 16).unwrap();
        assert_eq!(
            exact_relative_event_dates("I went to a castle last Friday.", session_date),
            vec![chrono::NaiveDate::from_ymd_opt(2023, 11, 10).unwrap()]
        );
        assert_eq!(
            exact_relative_event_dates("I went to a castle on Friday.", session_date),
            vec![chrono::NaiveDate::from_ymd_opt(2023, 11, 10).unwrap()]
        );
    }

    #[test]
    fn week_constraints_select_only_the_record_in_the_requested_window() {
        let items = vec![
            test_item(
                1,
                0.60,
                "Session date: 16 November 2023\nD18:1 Tim: Last week I was visiting the UK.",
            ),
            test_item(
                2,
                0.99,
                "Session date: 23 November 2023\nD19:1 Tim: Last week I was visiting the UK.",
            ),
        ];

        let second_week = build_query_plan(
            "Which country was Tim visiting in the second week of November?",
            &items,
        );
        assert_eq!(second_week.temporal_scope, "bounded");
        assert_eq!(second_week.temporal_granularity, TemporalGranularity::Week);
        assert_eq!(
            second_week.temporal_constraint.as_str(),
            "week_of_month:11:2"
        );
        assert!(
            second_week.object_terms.is_empty(),
            "{:?}",
            second_week.object_terms
        );
        assert!(
            second_week.quantity_terms.is_empty(),
            "{:?}",
            second_week.quantity_terms
        );
        assert!(second_week
            .required_slots
            .contains(&"temporal_constraint".to_string()));

        let pipeline = plan_answer_pipeline(
            "Which country was Tim visiting in the second week of November?",
            &items,
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[1],
            "{:#?}",
            pipeline.receipt
        );
        assert_eq!(
            pipeline
                .receipt
                .slot_coverage
                .iter()
                .find(|slot| slot.slot == "temporal_constraint")
                .map(|slot| slot.status.as_str()),
            Some("covered")
        );

        let preceding_week =
            build_query_plan("Where was Tim in the week before 16 November 2023?", &items);
        assert_eq!(preceding_week.temporal_scope, "bounded");
        assert_eq!(
            preceding_week.temporal_granularity,
            TemporalGranularity::Week
        );
        assert_eq!(
            preceding_week.temporal_constraint.as_str(),
            "week_before:2023-11-16"
        );
        assert!(preceding_week.object_terms.is_empty());

        let pipeline = plan_answer_pipeline(
            "Where was Tim in the week before 16 November 2023?",
            &items,
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[1],
            "{:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn person_location_requires_presence_not_a_topical_location_mention() {
        let items = vec![
            test_item(
                18,
                0.20,
                "Session date: 16 November 2023\nD18:1 Tim: I went to a castle during my trip to the UK last Friday.",
            ),
            test_item(
                19,
                0.99,
                "Session date: 21 November 2023\nD19:3 Tim: Last week I had a setback. I tried writing a story based on my experiences in the UK.",
            ),
        ];

        for query in [
            "Which country was Tim visiting in the second week of November?",
            "Where was Tim in the week before 16 November 2023?",
        ] {
            let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
            assert_eq!(
                pipeline.selected_memory_ids(),
                &[18],
                "{:#?}",
                pipeline.receipt
            );
            let draft = pipeline.unified_draft.expect("travel location answer");
            assert!(draft.text.contains("UK"), "{}", draft.text);
            assert!(
                !draft.text.contains("experiences in the UK"),
                "{}",
                draft.text
            );
        }

        let action_location = build_query_plan("Where did Tim write a story?", &[]);
        assert!(!location_query_requires_subject_presence(&action_location));
    }

    #[test]
    fn direct_presence_location_renderer_returns_a_concise_subject_location_claim() {
        let query = "Which country was Tim visiting?";
        let items = vec![test_item(
            18,
            0.9,
            "D18:1 Tim: I went to a castle during my trip to the UK last Friday.",
        )];
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[18],
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline.unified_draft.expect("direct location answer");
        assert!(draft.text.contains("Tim was in the UK."), "{}", draft.text);
        assert!(!draft.text.contains("castle"), "{}", draft.text);
        assert!(draft.text.contains("memory#18"), "{}", draft.text);
    }

    #[test]
    fn multi_entity_inference_preserves_premises_without_inventing_an_outcome() {
        let query =
            "Which Lunar Quest-related exhibits would Mira enjoy during her visit to Norway?";
        let items = vec![
            test_item(
                1,
                0.99,
                "D1:1 Mira: I went on a museum tour in London which I enjoyed.",
            ),
            test_item(2, 0.2, "D2:1 Mira: Lunar Quest is my favorite movie."),
            test_item(3, 0.2, "D3:1 Mira: I am visiting Norway next month."),
            test_item(
                4,
                0.99,
                "D4:1 Nora: I enjoy Lunar Quest exhibits in Norway.",
            ),
            test_item(5, 0.99, "D5:1 Mira: I enjoy exhibits about lunar science."),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 5, 5, None, "auto", 10, true);
        assert_eq!(
            pipeline.receipt.query_plan.named_entities,
            vec!["Lunar Quest", "Norway"]
        );
        assert_eq!(pipeline.receipt.query_plan.object_terms, vec!["exhibits"]);
        assert_eq!(
            pipeline
                .selected_memory_ids()
                .iter()
                .copied()
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([2, 3]),
            "{:#?}",
            pipeline.receipt
        );
        for slot in ["named_entity:lunarquest", "named_entity:norway"] {
            assert!(pipeline
                .receipt
                .slot_coverage
                .iter()
                .any(|row| row.slot == slot && row.status == "covered"));
        }
        assert!(!pipeline.evidence_atoms.iter().any(atom_is_answer_bearing));
        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("explicit partial with known premises");
        assert_eq!(draft.draft_mode, "partial");
        assert!(draft.text.contains("does not establish"), "{}", draft.text);
        assert!(draft.text.contains("favorite movie"), "{}", draft.text);
        assert!(draft.text.contains("visiting Norway"), "{}", draft.text);
        assert!(!draft.text.contains("London"), "{}", draft.text);
        assert_eq!(draft.rendered_claims.len(), 2);
        assert!(pipeline.operator_is_admissible(AnswerOperator::UnifiedGeneral));
        assert!(!pipeline.operator_is_admissible(AnswerOperator::GroundedInference));
        assert!(!pipeline.operator_is_admissible(AnswerOperator::GroundedList));
        assert!(pipeline
            .unified_draft_coverage()
            .unwrap()
            .iter()
            .all(|claim| claim.covered));
        for premise in &draft.rendered_claims {
            let text = premise.text.as_deref().unwrap();
            let ids = premise.evidence_ids.iter().copied().collect::<HashSet<_>>();
            assert!(pipeline.verify_claim_evidence(text, &ids).is_bound());
            for contaminated in [
                format!("{text} Therefore Mira will enjoy the Lunar Quest exhibit in Norway."),
                format!("Mira will enjoy the Lunar Quest exhibit in Norway. {text}"),
            ] {
                assert!(
                    !pipeline
                        .verify_claim_evidence(&contaminated, &ids)
                        .is_bound(),
                    "{contaminated}"
                );
            }
        }
    }

    #[test]
    fn multi_entity_inference_reports_missing_entity_and_respects_node_budget() {
        let query =
            "Which Lunar Quest-related exhibits would Mira enjoy during her visit to Norway?";
        let items = vec![test_item(
            2,
            0.8,
            "D2:1 Mira: Lunar Quest is my favorite movie.",
        )];
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        let draft = pipeline.unified_draft.as_ref().expect("one known premise");
        assert_eq!(draft.draft_mode, "partial");
        assert!(
            draft.text.contains("Missing context: Norway"),
            "{}",
            draft.text
        );
        assert!(pipeline
            .receipt
            .slot_coverage
            .iter()
            .any(|slot| slot.slot == "named_entity:norway" && slot.status == "missing"));
        let mut items = items;
        items.push(test_item(
            3,
            0.8,
            "D3:1 Mira: I am visiting Norway next month.",
        ));
        let limited = plan_answer_pipeline(query, &items, 2, 2, Some(1), "auto", 10, true);
        assert_eq!(limited.unified_draft.unwrap().rendered_claims.len(), 1);
    }

    #[test]
    fn multi_entity_inference_keeps_an_explicit_supported_outcome() {
        let query = "Which Lunar Quest-related exhibits would Mira enjoy in Norway?";
        let items = vec![test_item(
            7,
            0.8,
            "D7:1 Mira: I would enjoy Lunar Quest exhibits at the Aurora Museum in Norway.",
        )];
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        assert!(
            pipeline.evidence_atoms.iter().any(atom_is_answer_bearing),
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline.unified_draft.as_ref().expect("supported outcome");
        assert!(draft.text.contains("Aurora Museum"), "{}", draft.text);
        assert!(!draft.text.contains("does not establish"), "{}", draft.text);
        assert!(pipeline
            .unified_draft_coverage()
            .unwrap()
            .iter()
            .all(|claim| claim.covered));
    }

    #[test]
    fn multi_entity_capture_removes_question_scaffolding_and_keeps_distinct_keys() {
        assert_eq!(
            normalize_named_entity_capture("Which Lunar Quest-related"),
            Some("Lunar Quest".to_string())
        );
        assert_eq!(normalize_named_entity_capture("Which"), None);
        assert_ne!(
            named_entity_slot_name("Atlas 2"),
            named_entity_slot_name("Atlas 3")
        );
    }

    #[test]
    fn constrained_backoff_keeps_paraphrased_action_with_bound_subject_and_object() {
        let query = "What does John do to supplement his basketball training?";
        let items = vec![test_item(
            1,
            0.8,
            "D20:2 John: I am trying yoga for extra strength and flexibility alongside basketball training.",
        )];
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        let selected = pipeline
            .evidence_atoms
            .iter()
            .find(|atom| {
                atom.memory_id == 1 && atom.kind.grounding_role() == GroundingRole::Original
            })
            .expect("selected constrained fact");
        let action = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "action")
            .expect("action coverage");

        assert!(selected
            .supports_slots
            .contains(&"constrained_backoff_fact".to_string()));
        assert_eq!(action.status, "partial");
        assert!(pipeline
            .unified_draft
            .as_ref()
            .is_some_and(|draft| draft.text.contains("yoga")));
    }

    #[test]
    fn strategy_question_keeps_complementary_actions_from_separate_records() {
        let query = "What does John do to supplement his basketball training?";
        let items = vec![
            test_item(
                1,
                0.9,
                "D8:5 John: I use strength training alongside basketball training. It builds strength. I make time for it. It helps me stay balanced.",
            ),
            test_item(
                2,
                0.8,
                "D20:2 John: I am trying yoga to build extra strength and flexibility.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 4, true);

        assert_eq!(pipeline.receipt.query_plan.answer_shape, "action_set");
        assert!(pipeline.selected_memory_ids().contains(&1));
        assert!(pipeline.selected_memory_ids().contains(&2));
        assert!(pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 2
                && atom
                    .supports_slots
                    .contains(&"strategy_context_bridge".to_string())
        }));
        let answer = pipeline.unified_draft.expect("complementary action answer");
        assert!(answer.text.contains("strength training"), "{}", answer.text);
        assert!(answer.text.contains("yoga"), "{}", answer.text);
    }

    #[test]
    fn strategy_question_prefers_a_complementary_source_before_extra_turns_from_the_anchor() {
        let query = "What does John do to supplement his basketball training?";
        let items = vec![
            test_item(
                1,
                0.9,
                "D8:5 John: I use strength training alongside basketball training. It builds strength. I make time for it. It helps me stay balanced.\nD8:7 John: Strength training builds muscle and power for basketball.\nD8:9 John: Strength training improved my shooting accuracy, agility, and speed.",
            ),
            test_item(
                2,
                0.8,
                "D20:2 John: I'm also trying out yoga to get a little extra strength and flexibility. It's challenging but worth it.",
            ),
            test_item(
                3,
                0.8,
                "D27:14 John: Basketball is important to me, so I practice and train every day to stay in shape and improve.",
            ),
            test_item(
                4,
                0.8,
                "D14:3 John: I'm mentoring younger players on my team. It's also a great way to stay involved in the game during the off-season.",
            ),
            test_item(
                5,
                0.8,
                "D18:3 John: It was an intense season with both tough losses and great wins.",
            ),
            test_item(
                6,
                0.8,
                "D19:4 John: Finding that balance is key.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 3, 3, None, "auto", 4, true);

        assert!(pipeline.receipt.query_plan.requires_complementary_action);
        assert!(pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 1
                && atom
                    .supports_slots
                    .contains(&"complementary_action_anchor".to_string())
        }));
        assert!(pipeline.selected_memory_ids().contains(&1));
        assert!(
            pipeline.selected_memory_ids().contains(&2),
            "{:?}",
            pipeline.selected_memory_ids()
        );
        assert!(!pipeline.selected_memory_ids().contains(&4));
        assert!(!pipeline.selected_memory_ids().contains(&5));
        assert!(!pipeline.selected_memory_ids().contains(&6));
        let answer = pipeline.unified_draft.expect("complementary action answer");
        assert!(answer.text.contains("strength training"), "{}", answer.text);
        assert!(answer.text.contains("yoga"), "{}", answer.text);
    }

    #[test]
    fn aspect_set_binds_named_entity_on_the_fact_not_a_neighbor() {
        let query = "What does John like about LeBron James?";
        let items = vec![
            test_item(
                1,
                0.9,
                "D12:19 Tim: Who's your favorite basketball team/player?\nD12:20 John: The Wolves are my team for sure. And LeBron is the man - love his skills and leadership.",
            ),
            test_item(
                2,
                0.8,
                "D16:12 John: LeBron's moments of determination and heart are incredible. It's why I enjoy playing and pushing myself.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);

        assert_eq!(pipeline.receipt.query_plan.answer_shape, "aspect_set");
        let unified_position = pipeline
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::UnifiedGeneral)
            .expect("unified answer operator");
        let list_position = pipeline
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::GroundedList)
            .expect("grounded list fallback");
        assert!(unified_position < list_position);

        let answer_atoms = pipeline
            .evidence_atoms
            .iter()
            .filter(|atom| {
                atom.supports_slots
                    .contains(&"answer_bearing_fact".to_string())
            })
            .map(|atom| atom.text.as_str())
            .collect::<Vec<_>>();
        assert!(
            answer_atoms
                .iter()
                .any(|text| text.contains("skills and leadership")),
            "answer atoms: {answer_atoms:?}\nreceipt: {:#?}",
            pipeline.receipt
        );
        assert!(answer_atoms
            .iter()
            .any(|text| text.contains("determination and heart")));
        assert!(!answer_atoms.iter().any(|text| text.contains("The Wolves")));

        let answer = pipeline.unified_draft.expect("aspect-set answer");
        assert!(answer.text.contains("skills"), "{}", answer.text);
        assert!(answer.text.contains("leadership"), "{}", answer.text);
        assert!(answer.text.contains("determination"), "{}", answer.text);
        assert!(answer.text.contains("heart"), "{}", answer.text);
        assert!(!answer.text.contains("Wolves"), "{}", answer.text);
        assert!(
            !answer.text.contains("playing and pushing myself"),
            "{}",
            answer.text
        );
    }

    #[test]
    fn list_entry_class_bridges_a_preceding_question_across_media_atoms() {
        let query = "What fantasy movies does Tim like?";
        let items = vec![test_item(
            1,
            0.8,
            "D1:1 John: Do you have any favorite fantasy movies? [Image caption: a stack of discs]\nD1:2 Tim: Definitely Star Wars! It is my favorite and never gets old.",
        )];
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        let selected = pipeline
            .evidence_atoms
            .iter()
            .find(|atom| atom.text.contains("Definitely Star Wars"))
            .expect("selected movie answer");

        assert!(selected
            .supports_slots
            .contains(&"answer_class".to_string()));
        assert!(selected
            .supports_slots
            .contains(&"answer_bearing_fact".to_string()));
        assert!(pipeline
            .unified_draft
            .as_ref()
            .is_some_and(|draft| draft.text.contains("Star Wars")));
    }

    #[test]
    fn bounded_list_response_uses_nearby_same_speaker_category_context() {
        let query = "Which two fantasy novels does Tim particularly enjoy writing about?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.99,
                    "D9:5 Tim: I'm more into reading and fantasy novels. I love sinking into different magical worlds.",
                ),
                test_item(
                    2,
                    0.10,
                    "D4:5 Tim: I've been writing about different fantasy novels, studying characters, themes, and making book recommendations.\nD4:6 John: Do you have any favorite books you love writing about?\nD4:7 Tim: For sure! Harry Potter and Game of Thrones are amazing - I'm totally hooked!",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);

        assert_eq!(pipeline.receipt.query_plan.quantity_terms, vec!["two"]);
        assert_eq!(pipeline.receipt.query_plan.list_scope, "bounded");
        assert!(pipeline.selected_memory_ids().contains(&2));
        assert!(!pipeline.selected_memory_ids().contains(&1));
        let answer_bearing_atoms = pipeline
            .evidence_atoms
            .iter()
            .filter(|atom| atom_is_answer_bearing(atom))
            .collect::<Vec<_>>();
        assert_eq!(answer_bearing_atoms.len(), 1, "{answer_bearing_atoms:#?}");
        let selected = pipeline
            .evidence_atoms
            .iter()
            .find(|atom| atom.text.contains("Harry Potter and Game of Thrones"))
            .expect("selected bounded-list answer");
        assert!(selected
            .supports_slots
            .contains(&"bounded_list_response_bridge".to_string()));
        assert!(selected.supports_slots.contains(&"object".to_string()));
        assert!(selected
            .supports_slots
            .contains(&"answer_class".to_string()));
        assert_eq!(
            pipeline
                .receipt
                .slot_coverage
                .iter()
                .find(|slot| slot.slot == "list_scope")
                .map(|slot| slot.status.as_str()),
            Some("covered")
        );
        let draft = pipeline.unified_draft.expect("bounded list answer");
        assert!(
            draft.text.contains("Harry Potter and Game of Thrones"),
            "{}",
            draft.text
        );
        assert!(!draft.text.contains("sinking into different magical worlds"));
    }

    #[test]
    fn initiation_questions_require_a_start_event_not_generic_topical_experience() {
        let query = "How did John get introduced to basketball?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.99,
                    "D1:50 John: I love playing pro ball - it's a constant challenge and keeps me growing.",
                ),
                test_item(
                    2,
                    0.10,
                    "D2:61 Tim: How did you get into the game?\nD2:63 John: Basketball has been a part of my life ever since I was a kid.\nD2:64 John: I'd watch NBA games with my dad and dream of playing on those big courts.\nD2:65 John: When I turned ten, dad signed me up for a local league, and I've been playing ever since.",
                ),
            ],
        );
        let plan = build_query_plan(query, &items);
        let mut ranked_atoms = items
            .iter()
            .flat_map(|item| {
                project_source(ProjectionInput {
                    memory_id: item.id,
                    content: &item.content,
                    source: &item.source,
                    entity: item.entity.as_deref(),
                    score: item.score,
                    is_event: item.event_metadata.is_some(),
                })
            })
            .collect::<Vec<_>>();
        rank_evidence_atoms(&plan, &mut ranked_atoms);
        let ranked_start_event = ranked_atoms
            .iter()
            .find(|atom| atom.text.contains("dad signed me up for a local league"))
            .expect("projected initiation event");
        assert!(
            ranked_start_event
                .supports_slots
                .contains(&"initiation_event".to_string()),
            "{ranked_atoms:#?}"
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);

        assert_eq!(pipeline.receipt.query_plan.answer_target, "initiation");
        assert_eq!(pipeline.receipt.query_plan.answer_shape, "initiation");
        assert!(pipeline
            .receipt
            .query_plan
            .required_slots
            .contains(&"initiation_event".to_string()));
        assert!(
            pipeline.selected_memory_ids().contains(&2),
            "{:#?}",
            pipeline.receipt
        );
        assert!(!pipeline.selected_memory_ids().contains(&1));
        let start_event = pipeline
            .evidence_atoms
            .iter()
            .find(|atom| atom.text.contains("dad signed me up for a local league"))
            .expect("selected initiation event");
        assert!(atom_is_answer_bearing(start_event));
        assert!(start_event
            .supports_slots
            .contains(&"initiation_event".to_string()));
        assert_eq!(
            pipeline
                .receipt
                .slot_coverage
                .iter()
                .find(|slot| slot.slot == "initiation_event")
                .map(|slot| slot.status.as_str()),
            Some("covered")
        );
        let draft = pipeline.unified_draft.expect("initiation answer");
        assert!(
            draft
                .text
                .contains("When John turned ten, dad signed John up for a local league"),
            "{}",
            draft.text
        );
        assert!(!draft.text.contains("When I turned ten"), "{}", draft.text);
        assert!(!draft.text.contains("playing pro ball"), "{}", draft.text);
    }


    #[test]
    fn open_list_is_partial_without_a_closure_and_omits_referential_continuations() {
        let query = "What fantasy movies does Tim like?";
        let items = vec![test_item(
            1,
            0.8,
            "D1:1 John: Do you have any favorite fantasy movies?\nD1:2 Tim: Definitely Star Wars! It's my favorite and never gets old.",
        )];
        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "auto", 10, true);
        let list_scope = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "list_scope")
            .expect("list scope coverage");
        assert_eq!(list_scope.status, "partial");
        let draft = pipeline.unified_draft.expect("partial list answer");
        assert_eq!(draft.draft_mode, "partial");
        assert!(draft.text.contains("Star Wars"), "{}", draft.text);
        assert!(!draft.text.contains("It's my favorite"), "{}", draft.text);
        assert!(draft.text.contains("does not establish a complete list"));
        assert_eq!(pipeline.receipt.claim_plan.len(), 1);
        assert!(pipeline.receipt.claim_plan[0]
            .text
            .as_deref()
            .is_some_and(|text| text.contains("Star Wars")));
    }

    #[test]
    fn temporal_evidence_keeps_explicit_year_events_without_session_date_atom() {
        let query = "When did John take a trip to the Rocky Mountains?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(
                    1,
                    0.10,
                    "D20:40 John: In 2022, I took a trip to the Rocky Mountains.",
                ),
                test_item(2, 0.95, "D2:1 John: I enjoy hiking on weekends."),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert!(pipeline.selected_memory_ids().contains(&1));
        assert_eq!(
            pipeline
                .receipt
                .slot_coverage
                .iter()
                .find(|coverage| coverage.slot == "event_time")
                .map(|coverage| coverage.status.as_str()),
            Some("covered")
        );
        let draft = pipeline
            .unified_draft
            .expect("explicit-year temporal answer");
        assert!(draft.text.contains("2022"), "{}", draft.text);
    }

    #[test]
    fn location_questions_require_location_evidence_not_generic_activity() {
        let query = "Which city is John excited to have a game at?";
        let items = rerank_answer_candidates(
            query,
            &[
                test_item(1, 0.95, "D3:18 John: The stadium atmosphere feels great."),
                test_item(
                    2,
                    0.10,
                    "D3:19 John: It's Seattle, I'm stoked for my game there next month!",
                ),
            ],
        );
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert!(pipeline.selected_memory_ids().contains(&2));
        assert_eq!(
            pipeline
                .receipt
                .slot_coverage
                .iter()
                .find(|coverage| coverage.slot == "location")
                .map(|coverage| coverage.status.as_str()),
            Some("covered")
        );
        let draft = pipeline.unified_draft.expect("location answer");
        assert!(draft.text.contains("Seattle"), "{}", draft.text);
        assert!(!draft.text.contains("stadium atmosphere"), "{}", draft.text);
    }

    #[test]
    fn atom_level_plan_excludes_unselected_same_record_statements() {
        let query = "When did John get an ankle injury in 2023?";
        let items = vec![test_item(
            73,
            0.99,
            "Session date: 2023-11-20\nJohn: Last season, I had a major challenge when I hurt my ankle.\nJohn: I got an ankle injury on November 16, 2023.",
        )];

        let pipeline = plan_answer_pipeline(query, &items, 1, 1, None, "standard", 8, false);
        let selected_text = pipeline
            .evidence_atoms
            .iter()
            .map(|atom| atom.text.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        let scoped = pipeline.scoped_items(&items);
        let event_time = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "event_time")
            .expect("event-time coverage");

        assert!(selected_text.contains("November 16, 2023"));
        assert!(!selected_text.contains("Last season, I had a major challenge"));
        assert_eq!(event_time.status, "covered");
        assert_eq!(scoped.len(), 1);
        assert!(scoped[0].content.contains("November 16, 2023"));
        assert!(!scoped[0]
            .content
            .contains("Last season, I had a major challenge"));
    }

    #[test]
    fn temporal_selection_does_not_assign_session_date_to_retroactive_turn_fragments() {
        let query = "When did John get an ankle injury in 2023?";
        let items = vec![
            test_item(
                18,
                0.50,
                "Session date: 3:59 pm on 16 November, 2023\nD18:2 John: It's been a tough week for me with this injury.",
            ),
            test_item(
                19,
                0.99,
                "Session date: 10:22 am on 21 November, 2023\nD19:6 John: Last season, I had a major challenge when I hurt my ankle. It required some time off and physical therapy. It was frustrating because I couldn't play or help the team.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        let selected = pipeline
            .selected_memory_ids()
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let event_time = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "event_time")
            .expect("event-time coverage");

        assert!(selected.contains(&18), "{selected:?}");
        assert!(!selected.contains(&19), "{selected:?}");
        assert_eq!(event_time.evidence_ids, vec![18]);
        let draft = pipeline
            .unified_draft
            .expect("session-anchored injury date");
        assert!(draft.text.contains("16 November 2023"), "{}", draft.text);
        assert!(!draft.text.contains("21 November 2023"), "{}", draft.text);
    }


    #[test]
    fn month_year_session_evidence_does_not_answer_a_day_question() {
        let pipeline = plan_answer_pipeline(
            "What day did Tim start playing the violin?",
            &[test_item(
                1,
                0.8,
                "Session date: August 2023\nD21:13 Tim: I started playing the violin.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert!(pipeline.unified_draft.is_none(), "{:#?}", pipeline.receipt);
    }


    #[test]
    fn count_questions_plan_aggregation_before_scalar_rendering() {
        let query = "How many times did John injure his ankle?";
        let pipeline = plan_answer_pipeline(
            query,
            &[test_item(1, 0.8, "D1:1 John: I injured my ankle twice.")],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        let count_position = pipeline
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::CrossSourceEventCount)
            .expect("count operator");
        let unified_position = pipeline
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::UnifiedGeneral)
            .expect("unified operator");
        assert!(count_position < unified_position);
    }

    #[test]
    fn count_proof_is_not_limited_by_answer_display_nodes() {
        let query = "How many times did John injure his ankle?";
        let pipeline = plan_answer_pipeline(
            query,
            &[
                test_item(
                    31,
                    0.9,
                    "D31:1 John: I hurt my ankle while skiing on 10 May 2023.",
                ),
                test_item(
                    32,
                    0.8,
                    "D32:1 John: I injured my ankle during training on 12 May 2023.",
                ),
            ],
            2,
            2,
            Some(1),
            "concise",
            10,
            true,
        );
        let draft = pipeline.unified_draft.as_ref().expect("count answer");
        assert!(draft.text.contains("Direct answer: 2."), "{}", draft.text);
        assert!(draft.text.contains("memory#31"), "{}", draft.text);
        assert!(draft.text.contains("memory#32"), "{}", draft.text);
        let proof = pipeline.verify_claim_evidence(&draft.text, &HashSet::from([31, 32]));
        assert_eq!(proof.memory_ids, vec![31, 32], "{proof:#?}");
        assert!(proof.is_bound(), "{proof:#?}");
    }

    #[test]
    fn modifier_before_did_keeps_subject_as_action_agent() {
        let media_plan = build_query_plan(
            "What kind of picture did Tim share as part of their Harry Potter book collection?",
            &[],
        );
        assert_eq!(media_plan.subject.as_deref(), Some("Tim"));
        assert!(media_plan.subject_is_action_agent);

        let count_plan = build_query_plan("How many times did John injure his ankle?", &[]);
        assert_eq!(count_plan.subject.as_deref(), Some("John"));
        assert!(count_plan.subject_is_action_agent);
    }

    #[test]
    fn question_plan_extracts_subject_local_predicates_without_modal_scaffolding() {
        let items = [
            test_item(1, 0.8, "D1:1 Melanie: I painted a sunrise."),
            test_item(2, 0.8, "D1:2 Caroline: I moved here four years ago."),
        ];

        let painted = build_query_plan("When did Melanie paint a sunrise?", &items);
        assert_eq!(painted.subject.as_deref(), Some("Melanie"));
        assert!(painted.subject_is_action_agent);
        assert!(painted.action_terms.contains(&"paint".to_string()));

        let moved = build_query_plan("Where did Caroline move from 4 years ago?", &items);
        assert_eq!(moved.subject.as_deref(), Some("Caroline"));
        assert!(moved.subject_is_action_agent);
        assert!(moved.action_terms.contains(&"move".to_string()));
        assert!(!moved.object_terms.contains(&"ago".to_string()));
        assert!(!moved.object_terms.contains(&"years".to_string()));

        let art = build_query_plan("What kind of art does Caroline make?", &items);
        assert_eq!(art.subject.as_deref(), Some("Caroline"));
        assert!(art.subject_is_action_agent);
        assert!(art.action_terms.contains(&"make".to_string()));

        let counterfactual = build_query_plan(
            "Would Melanie be more interested in playing soccer or watching it?",
            &items,
        );
        assert!(!counterfactual
            .named_entities
            .iter()
            .any(|entity| entity.eq_ignore_ascii_case("Would Melanie")));
    }

    #[test]
    fn month_of_year_constraint_excludes_other_month_events() {
        let items = vec![
            test_item(
                10,
                0.99,
                "Session date: 7 July 2022\nD10:14 Melanie: I went camping on 4 July 2022.",
            ),
            test_item(
                6,
                0.10,
                "Session date: 14 June 2023\nD6:10 Melanie: I went camping on 10 June 2023.",
            ),
        ];
        let query = "When did Melanie go camping in June?";
        let plan = build_query_plan(query, &items);
        assert_eq!(plan.temporal_scope, "bounded");
        assert_eq!(plan.temporal_granularity, TemporalGranularity::Month);
        assert_eq!(plan.temporal_constraint.as_str(), "month_of_year:06");
        assert!(plan
            .required_slots
            .contains(&"temporal_constraint".to_string()));

        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);
        assert_eq!(
            pipeline.selected_memory_ids(),
            &[6],
            "{:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn household_possessive_question_requires_the_owner_s_direct_statement() {
        let query = "What do Melanie's kids like?";
        let items = vec![
            test_item(
                19,
                0.99,
                "D19:3 Caroline: Thanks so much, Melanie! I love helping kids.",
            ),
            test_item(
                11,
                0.98,
                "D11:7 Melanie: It is important to cultivate a loving and accepting environment for our kids.\nD11:11 Melanie: Your art is amazing, Caroline. I love how you use it to tell your stories.\nD11:13 Melanie: I love the self-acceptance and love theme.",
            ),
            test_item(
                13,
                0.97,
                "D13:10 Melanie: I would love to make a self-portrait someday.",
            ),
            test_item(
                8,
                0.96,
                "D8:6 Melanie: We love painting together, especially nature-inspired ones.",
            ),
            test_item(
                6,
                0.10,
                "D6:4 Melanie: Yesterday I took the kids to the museum.\nD6:6 Melanie: They were stoked for the dinosaur exhibit! They love learning about animals.",
            ),
            test_item(
                4,
                0.09,
                "D4:8 Melanie: The 2 younger kids love nature.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 4, 4, None, "auto", 10, true);
        assert_eq!(
            pipeline.receipt.query_plan.subject.as_deref(),
            Some("Melanie")
        );
        assert!(query_subject_is_action_agent(
            &query.to_lowercase(),
            pipeline.receipt.query_plan.subject.as_deref(),
        ));
        assert_eq!(pipeline.receipt.query_plan.cardinality, "list");
        assert!(pipeline.receipt.query_plan.household_dependent_preference);
        assert!(pipeline
            .receipt
            .query_plan
            .required_slots
            .contains(&"household_dependent_preference".to_string()));
        let selected_ids = pipeline
            .receipt
            .evidence_plan
            .selected_answer_atom_ids
            .iter()
            .collect::<HashSet<_>>();
        let selected_originals = pipeline
            .evidence_atoms
            .iter()
            .filter(|atom| {
                atom.kind.grounding_role() == GroundingRole::Original
                    && selected_ids.contains(&atom.atom_id)
            })
            .collect::<Vec<_>>();
        assert!(
            selected_originals.iter().any(|atom| atom.memory_id == 6),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            selected_originals.iter().any(|atom| atom.memory_id == 4),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !selected_originals.iter().any(|atom| atom.memory_id == 19),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !selected_originals.iter().any(|atom| atom.memory_id == 11),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !selected_originals
                .iter()
                .any(|atom| matches!(atom.memory_id, 8 | 13)),
            "{:#?}",
            pipeline.receipt
        );
        assert!(pipeline.operator_is_admissible(AnswerOperator::GroundedList));
        let list_evidence = pipeline.grounded_list_items(&items);
        let rendered_input = list_evidence
            .iter()
            .map(|item| item.content.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        for expected in ["dinosaur exhibit", "learning about animals", "nature"] {
            assert!(rendered_input.contains(expected), "{rendered_input}");
        }
        assert!(
            !rendered_input.contains("cultivate a loving"),
            "{rendered_input}"
        );
    }

    #[test]
    fn low_information_statement_rejects_phatic_acknowledgements() {
        assert!(is_low_information_statement("Sounds awesome!"));
        assert!(is_low_information_statement("That sounds awesome."));
        assert!(!is_low_information_statement(
            "Being surrounded by teammates who are equally passionate creates a strong bond."
        ));
    }
    #[test]
    fn media_provenance_question_prefers_attribution_over_generic_caption() {
        let query =
            "What kind of picture did Tim share as part of their Harry Potter book collection?";
        let pipeline = plan_answer_pipeline(
            query,
            &[
                test_item(
                    2,
                    0.20,
                    "D2:7 Tim: This is my book collection so far. [Image caption: a photo of a bookshelf with books and a picture on it]\nD2:8 John: That picture is really interesting. What's up with it?\nD2:9 Tim: That picture is from MinaLima. They created all the props for the Harry Potter films.",
                ),
                test_item(
                    22,
                    0.99,
                    "D22:8 John: Sounds awesome! Tim, what kind of stuff do they do in the Harry Potter series?",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );

        assert!(pipeline.receipt.query_plan.media_provenance_requested);
        assert!(
            pipeline.selected_memory_ids().contains(&2),
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline.unified_draft.expect("media provenance answer");
        assert!(draft.text.contains("MinaLima"), "{}", draft.text);
        assert!(
            !draft
                .text
                .contains("collection of Harry Potter books on a desk"),
            "{}",
            draft.text
        );
    }

    #[test]
    fn co_participant_event_uses_shared_event_phrase_as_the_answer() {
        let query = "What did Anthony and John end up playing during the charity event?";
        let pipeline = plan_answer_pipeline(
            query,
            &[test_item(
                4,
                0.20,
                "D4:8 John: Anthony and I went to this charity thing and ended up in this intense Harry Potter trivia contest.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );

        assert!(pipeline.receipt.query_plan.co_participant_event_requested);
        assert!(
            pipeline.selected_memory_ids().contains(&4),
            "{:#?}",
            pipeline.receipt
        );
        assert!(pipeline.evidence_atoms.iter().any(|atom| {
            atom.supports_slots
                .contains(&"co_participant_event".to_string())
        }));
        let draft = pipeline.unified_draft.as_ref().unwrap_or_else(|| {
            panic!(
                "co-participant event answer: {:#?}\natoms: {:#?}",
                pipeline.receipt, pipeline.evidence_atoms
            )
        });
        assert!(
            draft
                .text
                .contains("this intense Harry Potter trivia contest"),
            "{}",
            draft.text
        );
        assert!(pipeline
            .verify_claim_evidence(&draft.text, &HashSet::from([4]))
            .is_bound());
    }

    #[test]
    fn recommendation_combines_demonstrated_capability_and_preference() {
        let query = "What would be a good hobby related to his travel dreams for Tim to pick up?";
        let items = [
            test_item(
                4,
                0.20,
                "D4:1 Tim: I have been writing articles about the places I visit.",
            ),
            test_item(
                27,
                0.20,
                "D27:37 Tim: I love travel and dream of seeing the world.",
            ),
        ];
        let plan = build_query_plan(query, &items);
        assert!(plan.recommendation_requested);
        assert!(
            plan.recommendation_topic_terms
                .iter()
                .any(|term| term == "travel"),
            "{plan:#?}"
        );
        let mut atoms = items
            .iter()
            .flat_map(|item| {
                project_source(ProjectionInput {
                    memory_id: item.id,
                    content: &item.content,
                    source: &item.source,
                    entity: item.entity.as_deref(),
                    score: item.score,
                    is_event: false,
                })
            })
            .collect::<Vec<_>>();
        rank_evidence_atoms(&plan, &mut atoms);
        let draft = build_preference_recommendation_answer(&plan, &atoms, false)
            .expect("grounded recommendation");

        assert_eq!(draft.draft_mode, "inferred");
        assert!(
            draft.text.contains("Writing a travel blog"),
            "{}",
            draft.text
        );
        assert_eq!(
            draft.rendered_claims[0].evidence_ids,
            vec![4, 27],
            "{draft:#?}"
        );
    }
    #[test]
    fn likely_inference_is_confidence_labeled_and_claim_bound_to_each_basis() {
        let query = "What is likely to help John recover?";
        let items = [
            test_item(
                1,
                0.8,
                "D1:1 John: Physical therapy helped me recover after the injury.",
            ),
            test_item(
                2,
                0.8,
                "D2:4 John: Resting between practices also helped me recover.",
            ),
        ];
        let pipeline = plan_answer_pipeline(query, &items, 2, 2, None, "auto", 10, true);

        assert_eq!(
            pipeline.receipt.query_plan.inference_type,
            InferenceType::Likely.as_str()
        );
        let draft = pipeline.unified_draft.as_ref().expect("likely answer");
        assert_eq!(draft.draft_mode, "inferred", "{:#?}", pipeline.receipt);
        assert!(
            draft.text.contains("Likely answer (moderate confidence)"),
            "{}",
            draft.text
        );
        assert!(draft.text.contains("Physical therapy"), "{}", draft.text);
        assert!(
            draft.text.contains("Resting between practices"),
            "{}",
            draft.text
        );
        assert!(draft.text.contains("memory#1"), "{}", draft.text);
        assert!(draft.text.contains("memory#2"), "{}", draft.text);

        let coverage = pipeline
            .unified_draft_coverage()
            .expect("inference claims should bind their factual basis");
        assert!(
            coverage.iter().all(|report| report.covered),
            "{coverage:#?}"
        );
        assert_eq!(
            coverage
                .iter()
                .flat_map(|report| report.evidence_ids.iter().copied())
                .collect::<HashSet<_>>(),
            HashSet::from([1, 2])
        );
    }

    #[test]
    fn likely_inference_keeps_subject_binding_when_direct_object_gate_is_not_needed() {
        let pipeline = plan_answer_pipeline(
            "What is likely to help John recover?",
            &[test_item(
                1,
                0.9,
                "D1:1 Tim: Physical therapy helped me recover after the injury.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );
        assert!(pipeline.unified_draft.is_none(), "{:#?}", pipeline.receipt);
    }

    #[test]
    fn counterfactual_renderer_never_relabels_an_unobserved_outcome_as_fact() {
        let item = test_item(
            1,
            0.8,
            "D1:1 John: Physical therapy helped me recover after the injury.",
        );
        let plan = build_query_plan(
            "What would have happened if John had skipped physical therapy?",
            &[item.clone()],
        );
        assert_eq!(plan.inference_type, InferenceType::Counterfactual);
        let atoms = project_source(ProjectionInput {
            memory_id: item.id,
            content: &item.content,
            source: &item.source,
            entity: item.entity.as_deref(),
            score: item.score,
            is_event: false,
        });
        let selected = atoms.iter().collect::<Vec<_>>();
        let coverage = vec![AnswerSlotCoverageReceipt {
            slot: "inference_basis".to_string(),
            status: "partial".to_string(),
            evidence_ids: vec![1],
        }];
        let answer = build_grounded_inference_answer(&plan, &selected, &atoms, &coverage, false)
            .expect("counterfactual rendering");

        assert!(answer.starts_with("Counterfactual assessment (low confidence)"));
        assert!(answer.contains("not directly recorded"), "{answer}");
        assert!(!answer.starts_with("Direct answer:"), "{answer}");
    }

    #[test]
    fn full_space_records_require_an_explicit_chapter_selector() {
        assert!(!requires_full_space_answer_records(
            "What happened last year?"
        ));
        assert!(requires_full_space_answer_records(
            "What happened in Chapter 12?"
        ));
        assert!(requires_full_space_answer_records("第12章发生了什么？"));
        assert!(requires_full_space_answer_records("第一章发生了什么？"));
        assert!(!requires_full_space_answer_records("第一天发生了什么？"));
        assert!(!requires_full_space_answer_records("第十二节讲了什么？"));
        assert_eq!(
            extract_structural_selectors("第一百二十五章"),
            vec!["chapter:125"]
        );
        assert_eq!(extract_structural_selectors("第十二节"), vec!["section:12"]);
        assert_eq!(
            extract_structural_selectors("第一、三章"),
            vec!["chapter:1", "chapter:3"]
        );
        assert_eq!(
            extract_structural_selectors("第1、三章"),
            vec!["chapter:1", "chapter:3"]
        );
        assert_eq!(
            matching_structural_selectors("Section: 第1章", &["chapter:1".into()]),
            vec!["chapter:1"]
        );
    }

    #[test]
    fn question_plan_extracts_perception_action() {
        let plan = build_query_plan(
            "What did Caroline and her family see during their camping trip last year?",
            &[],
        );
        assert_eq!(plan.action_terms, vec!["see".to_string()]);
        assert!(
            !plan.object_terms.iter().any(|term| term == "family"),
            "{plan:#?}"
        );
        assert_eq!(plan.temporal_scope, "bounded");
        assert!(plan
            .required_slots
            .contains(&"relative_time_reference".to_string()));
    }

    #[test]
    fn object_query_rejects_directive_as_observed_action_fact() {
        let pipeline = plan_answer_pipeline(
            "What did Caroline research?",
            &[
                test_item(
                    1,
                    1.0,
                    "D17:1 Caroline: Do your research and find an adoption agency or lawyer.\nD17:2 Caroline: They'll help with the process and provide all the info.",
                ),
                test_item(2, 0.2, "D2:8 Caroline: I researched adoption agencies."),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );

        assert!(pipeline
            .receipt
            .query_plan
            .required_slots
            .contains(&"observed_action_fact".to_string()));
        assert!(!pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 1
                && atom
                    .supports_slots
                    .contains(&"answer_bearing_fact".to_string())
        }));
        assert!(pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 2
                && atom
                    .supports_slots
                    .contains(&"answer_bearing_fact".to_string())
        }));
        let observed_coverage = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "observed_action_fact")
            .expect("observed action fact coverage");
        assert_eq!(observed_coverage.status, "covered", "{pipeline:#?}");
        assert_eq!(observed_coverage.evidence_ids, vec![2], "{pipeline:#?}");

        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("observed action fact answer");
        assert!(draft.text.contains("adoption agencies"), "{}", draft.text);
        assert!(
            !draft.text.contains("find an adoption agency or lawyer"),
            "{}",
            draft.text
        );
    }

    #[test]
    fn object_query_rejects_bare_action_declaration_without_complement() {
        let pipeline = plan_answer_pipeline(
            "What did Caroline research?",
            &[
                test_item(1, 1.0, "D2:8 Caroline: I'm off to go do some research."),
                test_item(
                    2,
                    0.2,
                    "D2:9 Caroline: I've been researching adoption agencies.",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );

        assert!(pipeline
            .receipt
            .query_plan
            .required_slots
            .contains(&"action_object_complement".to_string()));
        assert!(!pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 1
                && atom
                    .supports_slots
                    .contains(&"answer_bearing_fact".to_string())
        }));
        assert!(pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 2
                && atom
                    .supports_slots
                    .contains(&"answer_bearing_fact".to_string())
        }));
        let complement_coverage = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "action_object_complement")
            .expect("action-object complement coverage");
        assert_eq!(complement_coverage.status, "covered", "{pipeline:#?}");
        assert_eq!(complement_coverage.evidence_ids, vec![2], "{pipeline:#?}");

        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("object answer from the substantive action fact");
        assert!(draft.text.contains("adoption agencies"), "{}", draft.text);
        assert!(
            !draft.text.contains("off to go do some research"),
            "{}",
            draft.text
        );
    }

    #[test]
    fn duration_queries_require_direct_duration_values() {
        let pipeline = plan_answer_pipeline(
            "How long has Caroline's current group of friends been together?",
            &[
                test_item(1, 1.0, "D3:11 Caroline: Yeah, I'm really lucky to have them."),
                test_item(
                    2,
                    0.2,
                    "D3:13 Caroline: I've known these friends for 4 years, since I moved from my home country.",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        assert_eq!(pipeline.receipt.query_plan.answer_target, "duration");
        assert!(pipeline
            .receipt
            .query_plan
            .required_slots
            .contains(&"duration_value".to_string()));
        let focused_position = pipeline
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::FocusedScalar)
            .expect("focused duration operator");
        let unified_position = pipeline
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::UnifiedGeneral)
            .expect("unified operator");
        assert!(focused_position < unified_position);
        assert!(
            pipeline.selected_memory_ids().contains(&2),
            "{:#?}",
            pipeline.receipt
        );
        assert!(!pipeline.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 1
                && atom
                    .supports_slots
                    .contains(&"answer_bearing_fact".to_string())
        }));
        let duration_value = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "duration_value")
            .expect("duration value coverage");
        assert_eq!(duration_value.status, "covered", "{:#?}", pipeline.receipt);
        assert_eq!(
            duration_value.evidence_ids,
            vec![2],
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline.unified_draft.expect("duration answer draft");
        assert!(draft.text.contains("4 years"), "{}", draft.text);
        assert!(!draft.text.contains("Yeah"), "{}", draft.text);
    }

    #[test]
    fn typed_grounded_lists_select_distinct_cross_record_values() {
        let pipeline = plan_answer_pipeline(
            "What activities does Melanie partake in?",
            &[
                test_item(1, 1.0, "D0:1 Melanie: I made some coffee before work."),
                test_item(2, 0.20, "D1:4 Melanie: I signed up for a pottery class."),
                test_item(
                    3,
                    0.19,
                    "D2:7 Melanie: My family and I went camping last weekend.",
                ),
                test_item(4, 0.18, "D3:3 Melanie: We love painting together."),
                test_item(5, 0.17, "D4:1 Melanie: I am going swimming with the kids."),
            ],
            5,
            5,
            None,
            "auto",
            10,
            true,
        );

        let plan = build_query_plan("What activities does Melanie partake in?", &[]);
        assert_eq!(
            plan.grounded_list_kind,
            Some(crate::memory_engine_summary::GroundedListKind::Activities),
            "{plan:#?}"
        );
        assert!(plan.required_slots.contains(&"list_value".to_string()));
        assert!(
            !plan.object_terms.iter().any(|term| term == "partake"),
            "{plan:#?}"
        );
        for expected in [2, 3, 4, 5] {
            assert!(
                pipeline.selected_memory_ids().contains(&expected),
                "{:#?}",
                pipeline.receipt
            );
        }
        assert!(
            !pipeline.selected_memory_ids().contains(&1),
            "{:#?}",
            pipeline.receipt
        );
        let list_value = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "list_value")
            .expect("typed list coverage");
        assert_eq!(list_value.status, "covered", "{:#?}", pipeline.receipt);
        assert_eq!(
            list_value
                .evidence_ids
                .iter()
                .copied()
                .collect::<HashSet<_>>(),
            HashSet::from([2, 3, 4, 5]),
            "{:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn typed_book_lists_retain_unnamed_references_for_partial_rendering() {
        let items = [
            test_item(
                1,
                1.0,
                "D1:1 Melanie: I loved reading \"Charlotte's Web\" as a child.",
            ),
            test_item(
                2,
                0.9,
                "D2:1 Melanie: This book I read last year reminded me to pursue my dreams.",
            ),
        ];
        let pipeline = plan_answer_pipeline(
            "What books has Melanie read?",
            &items,
            2,
            2,
            None,
            "auto",
            10,
            true,
        );

        assert!(
            pipeline.selected_memory_ids().contains(&1),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            pipeline.render_memory_ids().contains(&2),
            "{:#?}",
            pipeline.receipt
        );
        let list_items = pipeline.grounded_list_items(&items);
        assert_eq!(
            list_items
                .iter()
                .map(|item| item.id)
                .collect::<HashSet<_>>(),
            HashSet::from([1, 2]),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            list_items
                .iter()
                .any(|item| item.id == 2 && item.content.contains("This book I read last year")),
            "{list_items:#?}"
        );
    }

    #[test]
    fn household_grounded_lists_require_group_context() {
        let pipeline = plan_answer_pipeline(
            "What activities has Melanie done with her family?",
            &[
                test_item(1, 1.0, "D1:1 Melanie: I signed up for a pottery class."),
                test_item(
                    2,
                    0.20,
                    "D2:1 Melanie: My kids and I made pottery together.",
                ),
                test_item(3, 0.19, "D3:1 Melanie: My family and I went camping."),
                test_item(4, 0.18, "D4:1 Melanie: We painted together with the kids."),
                test_item(5, 0.17, "D5:1 Melanie: I took the kids to a museum."),
            ],
            5,
            5,
            None,
            "auto",
            10,
            true,
        );

        let plan = build_query_plan("What activities has Melanie done with her family?", &[]);
        assert!(plan.list_requires_household_context, "{plan:#?}");
        assert!(
            !plan.object_terms.iter().any(|term| term == "done"),
            "{plan:#?}"
        );
        for expected in [2, 3, 4, 5] {
            assert!(
                pipeline.selected_memory_ids().contains(&expected),
                "{:#?}",
                pipeline.receipt
            );
        }
        assert!(
            !pipeline.selected_memory_ids().contains(&1),
            "{:#?}",
            pipeline.receipt
        );
        let group_scope = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "list_participant_scope")
            .expect("household list scope coverage");
        assert_eq!(group_scope.status, "covered", "{:#?}", pipeline.receipt);
        assert_eq!(
            group_scope
                .evidence_ids
                .iter()
                .copied()
                .collect::<HashSet<_>>(),
            HashSet::from([2, 3, 4, 5]),
            "{:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn household_grounded_lists_reject_family_mentions_unrelated_to_activity() {
        let pipeline = plan_answer_pipeline(
            "What activities has Melanie done with her family?",
            &[
                test_item(
                    1,
                    1.0,
                    "D1:1 Melanie: I make time each day for running, reading, or playing my violin, which helps me stay present for my fam.",
                ),
                test_item(
                    2,
                    0.2,
                    "D2:1 Melanie: I went swimming with the kids last weekend.",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );

        assert!(
            !pipeline.selected_memory_ids().contains(&1),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            pipeline.selected_memory_ids().contains(&2),
            "{:#?}",
            pipeline.receipt
        );
        let participant_scope = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "list_participant_scope")
            .expect("family scope coverage");
        assert_eq!(
            participant_scope.evidence_ids,
            vec![2],
            "{:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn household_grounded_lists_allow_immediate_household_question_bridge() {
        let pipeline = plan_answer_pipeline(
            "What activities has Melanie done with her family?",
            &[test_item(
                1,
                1.0,
                "D1:1 Caroline: What other projects do you do with the kids?\nD1:2 Melanie: We love painting together.",
            )],
            1,
            1,
            None,
            "auto",
            10,
            true,
        );

        assert!(
            pipeline.selected_memory_ids().contains(&1),
            "{:#?}",
            pipeline.receipt
        );
        let participant_scope = pipeline
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "list_participant_scope")
            .expect("household bridge coverage");
        assert_eq!(
            participant_scope.evidence_ids,
            vec![1],
            "{:#?}",
            pipeline.receipt
        );
    }

    #[test]
    fn categorical_attributes_require_direct_subject_values() {
        let identity = plan_answer_pipeline(
            "What is Caroline's identity?",
            &[
                test_item(
                    1,
                    1.0,
                    "D13:1 Caroline: My art explores gender identity.\nD13:2 Caroline: Thanks, Mel!",
                ),
                test_item(2, 0.2, "D1:5 Caroline: I am a transgender woman."),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        assert!(identity
            .receipt
            .query_plan
            .required_slots
            .contains(&"categorical_attribute_value".to_string()));
        let categorical_position = identity
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::CategoricalAttribute)
            .expect("categorical operator");
        let unified_position = identity
            .operators()
            .iter()
            .position(|operator| *operator == AnswerOperator::UnifiedGeneral)
            .expect("unified operator");
        assert!(categorical_position < unified_position);
        assert!(
            identity.selected_memory_ids().contains(&2),
            "{:#?}",
            identity.receipt
        );
        assert!(!identity.evidence_atoms.iter().any(|atom| {
            atom.memory_id == 1
                && atom
                    .supports_slots
                    .contains(&"answer_bearing_fact".to_string())
        }));
        let identity_value = identity
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "categorical_attribute_value")
            .expect("identity categorical coverage");
        assert_eq!(identity_value.status, "covered", "{:#?}", identity.receipt);
        assert_eq!(
            identity_value.evidence_ids,
            vec![2],
            "{:#?}",
            identity.receipt
        );
        let identity_draft = identity.unified_draft.expect("identity answer draft");
        assert!(
            identity_draft
                .text
                .to_lowercase()
                .contains("transgender woman"),
            "{}",
            identity_draft.text
        );
        assert!(
            !identity_draft.text.contains("Thanks"),
            "{}",
            identity_draft.text
        );

        let relationship_status = plan_answer_pipeline(
            "What is Caroline's relationship status?",
            &[
                test_item(
                    3,
                    1.0,
                    "D13:3 Caroline: Relationships can be complicated. Thanks, Mel!",
                ),
                test_item(
                    4,
                    0.2,
                    "D2:14 Caroline: It will be tough as a single parent.",
                ),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        let status_value = relationship_status
            .receipt
            .slot_coverage
            .iter()
            .find(|slot| slot.slot == "categorical_attribute_value")
            .expect("relationship-status categorical coverage");
        assert_eq!(
            status_value.status, "covered",
            "{:#?}",
            relationship_status.receipt
        );
        assert_eq!(
            status_value.evidence_ids,
            vec![4],
            "{:#?}",
            relationship_status.receipt
        );
        let status_draft = relationship_status
            .unified_draft
            .expect("relationship-status answer draft");
        assert!(status_draft.text.to_lowercase().contains("single parent"));
        assert!(!status_draft
            .text
            .contains("Relationships can be complicated"));
    }

    #[test]
    fn collective_participant_event_prefers_local_group_fact() {
        let pipeline = plan_answer_pipeline(
            "What did Caroline and her family see during their camping trip last year?",
            &[
                test_item(1, 0.90, "D10:14 Melanie: I'll always remember our camping trip last year when we saw the Perseid meteor shower."),
                test_item(2, 0.90, "D18:1 Caroline: It's so sweet to see your love for your family, Melanie."),
                test_item(3, 1.00, "D16:1 Caroline: I had a wicked day out with the gang last weekend - we went biking and saw some pretty cool stuff.\nD16:2 Melanie: We went camping with the kids a few weeks ago."),
            ],
            2,
            2,
            None,
            "auto",
            10,
            true,
        );
        assert!(
            pipeline.selected_memory_ids().contains(&1),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !pipeline.selected_memory_ids().contains(&2),
            "{:#?}",
            pipeline.receipt
        );
        assert!(
            !pipeline.evidence_atoms.iter().any(|atom| {
                atom.memory_id == 3
                    && atom
                        .supports_slots
                        .contains(&"answer_bearing_fact".to_string())
            }),
            "{:#?}",
            pipeline.receipt
        );
        let draft = pipeline
            .unified_draft
            .as_ref()
            .expect("collective partial answer");
        assert_eq!(draft.draft_mode, "partial");
        assert!(draft.text.contains("Melanie"), "{}", draft.text);
        assert!(draft.text.contains("not Caroline"), "{}", draft.text);
        assert!(
            draft.text.contains("Perseid meteor shower"),
            "{}",
            draft.text
        );
        assert_eq!(draft.rendered_claims[0].evidence_ids, vec![1], "{draft:#?}");
        let coverage = pipeline
            .unified_draft_coverage()
            .expect("collective partial claim should retain the factual source");
        assert_eq!(coverage[0].evidence_ids, vec![1], "{coverage:#?}");
    }
}
