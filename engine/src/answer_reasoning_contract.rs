//! Shared, source-bound contracts for model planning and independent verification.
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) const CONTRACT_VERSION: &str = "answer_reasoning_v7_opaque_event_digests";

pub(crate) use crate::support::AnswerReasoningMode as ReasoningMode;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct SourceSpan {
    pub atom_id: String,
    /// UTF-8 byte offsets within the immutable atom body, not character offsets.
    pub start: usize,
    pub end: usize,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct SourceAtom {
    pub atom_id: String,
    pub memory_id: u64,
    pub source_version: u32,
    pub source_digest: String,
    pub body: String,
    pub modality: String,
    pub evidence_kind: String,
    pub speaker: Option<String>,
    pub context_group: usize,
    /// Captions are descriptions, not observations of the original media.
    pub descriptive_only: bool,
    pub source_anchor: Option<crate::api_types::AnswerEvidenceSourceAnchor>,
}

pub(crate) fn validate_span(
    span: &SourceSpan,
    atoms: &BTreeMap<String, SourceAtom>,
) -> Result<(), String> {
    let atom = atoms.get(&span.atom_id).ok_or("unknown_source_atom")?;
    if span.start >= span.end
        || span.text.is_empty()
        || atom.body.get(span.start..span.end) != Some(span.text.as_str())
    {
        // Audit detail: which quoted text failed against which atom.
        let quoted = span.text.chars().take(60).collect::<String>();
        return Err(format!("source_span_mismatch:{}:{quoted:?}", span.atom_id));
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct BoundValue {
    pub value: String,
    pub spans: Vec<SourceSpan>,
}

impl BoundValue {
    fn validate(&self, atoms: &BTreeMap<String, SourceAtom>) -> Result<(), String> {
        if self.value.trim().is_empty() || self.spans.is_empty() {
            return Err("unbound_relation_value".into());
        }
        for span in &self.spans {
            validate_span(span, atoms)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Role {
    Agent,
    Patient,
    Recipient,
    Possessor,
    Experiencer,
    Instrument,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Polarity {
    Positive,
    Negative,
    Unknown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Modality {
    Observed,
    Reported,
    Possible,
    Intended,
    Required,
    Counterfactual,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum ConditionExpr {
    Relation {
        relation_id: String,
    },
    All {
        args: Vec<Self>,
    },
    Any {
        args: Vec<Self>,
    },
    Not {
        arg: Box<Self>,
    },
    If {
        premise: Box<Self>,
        consequence: Box<Self>,
    },
    OnlyIf {
        consequence: Box<Self>,
        requirement: Box<Self>,
    },
    Unless {
        exception: Box<Self>,
        consequence: Box<Self>,
    },
    When {
        trigger: Box<Self>,
        consequence: Box<Self>,
    },
    Before {
        first: Box<Self>,
        second: Box<Self>,
    },
    After {
        first: Box<Self>,
        second: Box<Self>,
    },
    Cause {
        cause: Box<Self>,
        effect: Box<Self>,
    },
    ForAll {
        variable: String,
        domain: String,
        body: Box<Self>,
    },
    Exists {
        variable: String,
        domain: String,
        body: Box<Self>,
    },
    Unknown {
        reason: String,
    },
}

impl ConditionExpr {
    pub(crate) fn validate(&self, relations: &BTreeSet<String>) -> Result<(), String> {
        self.referenced_relations(relations).map(|_| ())
    }

    /// Only validated, complete trees may contribute proof dependencies.
    pub(crate) fn referenced_relations(
        &self,
        relations: &BTreeSet<String>,
    ) -> Result<BTreeSet<String>, String> {
        let mut stack = vec![(self, 1usize)];
        let mut count = 0;
        let mut referenced = BTreeSet::new();
        while let Some((node, depth)) = stack.pop() {
            count += 1;
            if depth > 16 || count > 256 {
                return Err(format!(
                    "condition_budget_exceeded:depth={depth}:visited_nodes={count}:pending_subtrees={}",
                    stack.len() + 1
                ));
            }
            let mut add_pair = |left, right| {
                stack.push((left, depth + 1));
                stack.push((right, depth + 1));
            };
            match node {
                Self::Relation { relation_id } => {
                    if !relations.contains(relation_id) {
                        return Err("unknown_condition_relation".into());
                    }
                    referenced.insert(relation_id.clone());
                }
                Self::All { args } | Self::Any { args } => {
                    if args.len() < 2 {
                        return Err("invalid_condition_arity".into());
                    }
                    for arg in args {
                        stack.push((arg, depth + 1));
                    }
                }
                Self::Not { arg } => stack.push((arg, depth + 1)),
                Self::If {
                    premise,
                    consequence,
                } => add_pair(premise.as_ref(), consequence.as_ref()),
                Self::OnlyIf {
                    consequence,
                    requirement,
                } => add_pair(consequence.as_ref(), requirement.as_ref()),
                Self::Unless {
                    exception,
                    consequence,
                } => add_pair(exception.as_ref(), consequence.as_ref()),
                Self::When {
                    trigger,
                    consequence,
                } => add_pair(trigger.as_ref(), consequence.as_ref()),
                Self::Before { first, second } | Self::After { first, second } => {
                    add_pair(first.as_ref(), second.as_ref())
                }
                Self::Cause { cause, effect } => add_pair(cause.as_ref(), effect.as_ref()),
                Self::ForAll {
                    variable,
                    domain,
                    body,
                }
                | Self::Exists {
                    variable,
                    domain,
                    body,
                } => {
                    if variable.trim().is_empty() || domain.trim().is_empty() {
                        return Err("unbound_quantifier".into());
                    }
                    stack.push((body, depth + 1));
                }
                Self::Unknown { .. } => return Err("condition_unresolved".into()),
            }
        }
        Ok(referenced)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct Relation {
    pub relation_id: String,
    pub predicate: BoundValue,
    pub roles: BTreeMap<Role, BoundValue>,
    pub time: Option<BoundValue>,
    pub place: Option<BoundValue>,
    pub polarity: Polarity,
    pub modality: Modality,
    pub condition: Option<ConditionExpr>,
}

impl Relation {
    pub(crate) fn validate(
        &self,
        atoms: &BTreeMap<String, SourceAtom>,
        relation_ids: &BTreeSet<String>,
    ) -> Result<(), String> {
        if self.relation_id.is_empty() || self.roles.is_empty() {
            return Err("incomplete_relation".into());
        }
        self.predicate.validate(atoms)?;
        for value in self
            .roles
            .values()
            .chain(self.time.iter())
            .chain(self.place.iter())
        {
            value.validate(atoms)?;
        }
        if self.polarity == Polarity::Unknown || self.modality == Modality::Unknown {
            return Err("relation_unresolved".into());
        }
        if let Some(condition) = &self.condition {
            condition.validate(relation_ids)?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct EventObservation {
    pub observation_id: String,
    pub relation_id: String,
    pub native_event_id: Option<BoundValue>,
    pub native_event_namespace: Option<BoundValue>,
    pub multiplicity: u64,
    pub quantity_source: Option<SourceSpan>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum IdentityKind {
    Same,
    Different,
    Unresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct IdentityLink {
    pub left: String,
    pub right: String,
    pub kind: IdentityKind,
    pub basis: Vec<SourceSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct EventGraph {
    pub observations: Vec<EventObservation>,
    pub links: Vec<IdentityLink>,
}

fn relations_conflict(left: &Relation, right: &Relation) -> bool {
    if left.predicate.value != right.predicate.value
        || left.polarity != right.polarity
        || left.modality != right.modality
        || left.condition != right.condition
    {
        return true;
    }
    for (role, value) in &left.roles {
        if right
            .roles
            .get(role)
            .is_some_and(|other| value.value != other.value)
        {
            return true;
        }
    }
    left.time
        .as_ref()
        .zip(right.time.as_ref())
        .is_some_and(|(a, b)| a.value != b.value)
        || left
            .place
            .as_ref()
            .zip(right.place.as_ref())
            .is_some_and(|(a, b)| a.value != b.value)
}

impl EventGraph {
    /// Called only after independent semantic verification of observations and links.
    /// Unknown identity never implies distinct occurrences.
    

    pub(crate) fn checked_bounds(
        &self,
        relations: &BTreeMap<String, Relation>,
        atoms: &BTreeMap<String, SourceAtom>,
    ) -> Result<EventCountBounds, String> {
        if self.observations.is_empty() {
            return Err("count_scope_unproven".into());
        }
        if self.observations.len() > 256 || self.links.len() > 65_536 {
            return Err("event_budget_exceeded".into());
        }
        let mut ids = BTreeMap::new();
        let relation_ids = relations.keys().cloned().collect();
        for (index, observation) in self.observations.iter().enumerate() {
            if observation.observation_id.is_empty()
                || ids
                    .insert(observation.observation_id.clone(), index)
                    .is_some()
            {
                return Err("duplicate_event_observation".into());
            }
            let relation = relations
                .get(&observation.relation_id)
                .ok_or("unknown_event_relation")?;
            relation.validate(atoms, &relation_ids)?;
            if relation.polarity != Polarity::Positive
                || !matches!(relation.modality, Modality::Observed | Modality::Reported)
            {
                return Err("event_not_observed".into());
            }
            match (
                &observation.native_event_id,
                &observation.native_event_namespace,
            ) {
                (Some(native), Some(namespace)) => {
                    for value in [native, namespace] {
                        value.validate(atoms)?;
                        if !value.spans.iter().any(|span| span.text == value.value) {
                            return Err("native_identity_not_literal".into());
                        }
                    }
                }
                (None, None) => {}
                _ => return Err("native_identity_namespace_missing".into()),
            }
            if observation.multiplicity == 0 {
                return Err("zero_occurrence_is_not_an_event".into());
            }
            if observation.multiplicity != 1 && observation.quantity_source.is_none() {
                return Err("unbound_event_quantity".into());
            }
            if let Some(span) = &observation.quantity_source {
                validate_span(span, atoms)?;
            }
        }
        let mut groups = (0..self.observations.len()).collect::<Vec<_>>();
        let mut different = BTreeSet::new();
        // Native identity is scoped, source-bound, and subject to independent
        // observation judgment. Never compare bare IDs across namespaces.
        for left in 0..self.observations.len() {
            for right in left + 1..self.observations.len() {
                let a = &self.observations[left];
                let b = &self.observations[right];
                if let (Some(a_id), Some(a_ns), Some(b_id), Some(b_ns)) = (
                    &a.native_event_id,
                    &a.native_event_namespace,
                    &b.native_event_id,
                    &b.native_event_namespace,
                ) {
                    if a_ns.value != b_ns.value {
                        continue;
                    }
                    if a_id.value == b_id.value {
                        let old = groups[right];
                        let new = groups[left];
                        for group in &mut groups {
                            if *group == old {
                                *group = new;
                            }
                        }
                    } else {
                        different.insert((left, right));
                    }
                }
            }
        }
        for link in &self.links {
            let left = *ids.get(&link.left).ok_or("unknown_event_link")?;
            let right = *ids.get(&link.right).ok_or("unknown_event_link")?;
            if left == right || link.basis.is_empty() {
                return Err("invalid_event_link".into());
            }
            for span in &link.basis {
                validate_span(span, atoms)?;
            }
            match link.kind {
                IdentityKind::Same => {
                    let old = groups[right];
                    let new = groups[left];
                    for group in &mut groups {
                        if *group == old {
                            *group = new;
                        }
                    }
                }
                IdentityKind::Different => {
                    different.insert((left.min(right), left.max(right)));
                }
                IdentityKind::Unresolved => {}
            }
        }
        for left in 0..self.observations.len() {
            for right in left + 1..self.observations.len() {
                if groups[left] != groups[right] {
                    continue;
                }
                let a = &self.observations[left];
                let b = &self.observations[right];
                if different.contains(&(left, right))
                    || a.multiplicity != b.multiplicity
                    || relations_conflict(&relations[&a.relation_id], &relations[&b.relation_id])
                {
                    return Err("event_cluster_conflict".into());
                }
            }
        }
        let unique = groups.iter().copied().collect::<BTreeSet<_>>();
        // Project distinctness onto merged groups once. Scanning all links for
        // each group pair made dense provider output need quartic work.
        let group_edges = different
            .iter()
            .map(|&(a, b)| (groups[a].min(groups[b]), groups[a].max(groups[b])))
            .collect::<BTreeSet<_>>();
        let complete = group_edges.len() == unique.len() * (unique.len() - 1) / 2;
        let exact = if complete {
            Some(unique.iter().try_fold(0u64, |sum, group| {
                let index = groups
                    .iter()
                    .position(|candidate| candidate == group)
                    .unwrap();
                sum.checked_add(self.observations[index].multiplicity)
                    .ok_or_else(|| "event_count_overflow".to_string())
            })?)
        } else {
            None
        };
        // When identity is unresolved, even distinct source observations may
        // describe the same occurrence. One largest observed group is a sound
        // lower bound, without inventing independent times or summing records.
        let lower_bound = if let Some(total) = exact {
            total
        } else {
            self.observations
                .iter()
                .map(|o| o.multiplicity)
                .max()
                .unwrap_or(0)
        };
        Ok(EventCountBounds { exact, lower_bound })
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct EventCountBounds {
    pub exact: Option<u64>,
    pub lower_bound: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CountKind {
    Exact,
    LowerBound,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct CountDerivation {
    pub event_set_id: String,
    pub value: u64,
    pub kind: CountKind,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct EventSet {
    pub event_set_id: String,
    pub graph: EventGraph,
    pub explicit_zero: Option<ZeroCountProof>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ZeroCountProof {
    pub relation_ids: Vec<String>,
    pub spans: Vec<SourceSpan>,
}

impl EventSet {
    /// Zero is an explicit scoped absence assertion, never an empty retrieval
    /// result or an observation with zero multiplicity. Semantic scope and the
    /// absence assertion itself still require independent verification.
    pub(crate) fn checked_bounds(
        &self,
        relations: &BTreeMap<String, Relation>,
        atoms: &BTreeMap<String, SourceAtom>,
    ) -> Result<EventCountBounds, String> {
        let Some(zero) = &self.explicit_zero else {
            return self.graph.checked_bounds(relations, atoms);
        };
        if !self.graph.observations.is_empty() || !self.graph.links.is_empty() {
            return Err("zero_proof_conflicts_with_event_graph".into());
        }
        if zero.relation_ids.is_empty()
            || zero.spans.is_empty()
            || zero.relation_ids.iter().collect::<BTreeSet<_>>().len() != zero.relation_ids.len()
        {
            return Err("unbound_zero_count_proof".into());
        }
        let ids = relations.keys().cloned().collect();
        for id in &zero.relation_ids {
            let relation = relations.get(id).ok_or("unknown_zero_count_relation")?;
            relation.validate(atoms, &ids)?;
            if relation.polarity != Polarity::Negative
                || !matches!(relation.modality, Modality::Observed | Modality::Reported)
            {
                return Err("zero_count_not_observed_absence".into());
            }
        }
        for span in &zero.spans {
            validate_span(span, atoms)?;
        }
        Ok(EventCountBounds {
            exact: Some(0),
            lower_bound: 0,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClaimConditionBinding {
    /// Source-relation AST proposed to qualify this conclusion.
    pub condition: ConditionExpr,
    /// Exact frozen question AST when the claim is constrained by the question.
    pub question_condition: Option<ConditionExpr>,
    pub inherited_from_question: bool,
    pub source_spans: Vec<SourceSpan>,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProposedClaim {
    pub claim_id: String,
    pub text: String,
    pub atom_ids: Vec<String>,
    pub relation_ids: Vec<String>,
    pub inferred: bool,
    pub count: Option<CountDerivation>,
    pub condition_binding: Option<ClaimConditionBinding>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Verdict {
    Supported,
    Contradicted,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(crate) struct ClaimJudgment {
    pub claim_id: String,
    pub verdict: Verdict,
    pub atom_ids: Vec<String>,
    pub roles: Verdict,
    pub predicate: Verdict,
    pub polarity: Verdict,
    pub modality: Verdict,
    pub conditions: Verdict,
    pub time: Verdict,
    pub quantities: Verdict,
    pub source_scope: Verdict,
    pub reason: String,
}

pub(crate) fn checked_claim_atoms(
    claim: &ProposedClaim,
    judgment: &ClaimJudgment,
    atoms: &BTreeMap<String, SourceAtom>,
) -> Result<Vec<SourceAtom>, String> {
    if claim.claim_id != judgment.claim_id
        || claim.text.trim().is_empty()
        || claim.atom_ids.is_empty()
    {
        return Err("invalid_claim_binding".into());
    }
    let failed = [
        ("verdict", judgment.verdict),
        ("roles", judgment.roles),
        ("predicate", judgment.predicate),
        ("polarity", judgment.polarity),
        ("modality", judgment.modality),
        ("conditions", judgment.conditions),
        ("time", judgment.time),
        ("quantities", judgment.quantities),
        ("source_scope", judgment.source_scope),
    ]
    .iter()
    .filter(|(_, status)| *status != Verdict::Supported)
    .map(|(name, status)| format!("{name}={}", if *status == Verdict::Contradicted { "contradicted" } else { "unknown" }))
    .collect::<Vec<_>>();
    if !failed.is_empty() {
        // The failing dimensions are part of the audit trail.
        return Err(format!("claim_not_supported:{}", failed.join(",")));
    }
    let planned = claim.atom_ids.iter().collect::<BTreeSet<_>>();
    let judged = judgment.atom_ids.iter().collect::<BTreeSet<_>>();
    if planned != judged
        || planned.len() != claim.atom_ids.len()
        || judged.len() != judgment.atom_ids.len()
    {
        return Err("claim_source_set_changed".into());
    }
    planned
        .into_iter()
        .map(|id| {
            atoms
                .get(id)
                .cloned()
                .ok_or_else(|| "unknown_claim_atom".into())
        })
        .collect()
}




#[cfg(test)]
mod tests {
    use super::*;

    fn atom() -> SourceAtom {
        SourceAtom {
            atom_id: "m1:a0".into(),
            memory_id: 1,
            source_version: 1,
            source_digest: "digest".into(),
            body: "甲 inspected twice.".into(),
            modality: "text".into(),
            evidence_kind: "body_fact".into(),
            speaker: None,
            context_group: 0,
            descriptive_only: false,
            source_anchor: None,
        }
    }
    fn span() -> SourceSpan {
        SourceSpan {
            atom_id: "m1:a0".into(),
            start: 0,
            end: 3,
            text: "甲".into(),
        }
    }
    fn bound(value: &str) -> BoundValue {
        BoundValue {
            value: value.into(),
            spans: vec![span()],
        }
    }
    fn relation(id: &str, actor: &str) -> Relation {
        Relation {
            relation_id: id.into(),
            predicate: bound("inspect"),
            roles: BTreeMap::from([(Role::Agent, bound(actor))]),
            time: None,
            place: None,
            polarity: Polarity::Positive,
            modality: Modality::Observed,
            condition: None,
        }
    }
    #[test]
    fn spans_require_exact_utf8_boundaries_and_source() {
        let atoms = BTreeMap::from([("m1:a0".into(), atom())]);
        assert!(validate_span(&span(), &atoms).is_ok());
        let mut bad = span();
        bad.end = 1;
        assert!(validate_span(&bad, &atoms).is_err());
        bad.end = 3;
        bad.text = "乙".into();
        assert!(validate_span(&bad, &atoms).is_err());
        bad = span();
        bad.atom_id = "other".into();
        assert!(validate_span(&bad, &atoms).is_err());
    }
    #[test]
    fn condition_direction_scope_and_limits_are_explicit() {
        let ids = BTreeSet::from(["a".into(), "b".into()]);
        let a = ConditionExpr::Relation {
            relation_id: "a".into(),
        };
        let b = ConditionExpr::Relation {
            relation_id: "b".into(),
        };
        let conditional = ConditionExpr::If {
            premise: Box::new(a.clone()),
            consequence: Box::new(b.clone()),
        };
        assert!(conditional.validate(&ids).is_ok());
        assert_ne!(
            conditional,
            ConditionExpr::OnlyIf {
                consequence: Box::new(a.clone()),
                requirement: Box::new(b)
            }
        );
        let mut nested = a;
        for _ in 0..16 {
            nested = ConditionExpr::Not {
                arg: Box::new(nested),
            };
        }
        assert!(nested
            .validate(&ids)
            .unwrap_err()
            .starts_with("condition_budget_exceeded:depth=17:"));
        assert!(ConditionExpr::Unknown {
            reason: "ambiguous".into()
        }
        .validate(&ids)
        .is_err());
    }
    #[test]
    fn condition_node_budget_never_returns_a_partial_dependency_set() {
        let ids = BTreeSet::from(["a".into()]);
        let leaf = ConditionExpr::Relation {
            relation_id: "a".into(),
        };
        let at_limit = ConditionExpr::All {
            args: vec![leaf.clone(); 255],
        };
        assert_eq!(at_limit.referenced_relations(&ids).unwrap(), ids);
        let over_limit = ConditionExpr::All {
            args: vec![leaf; 256],
        };
        assert_eq!(
            over_limit.referenced_relations(&ids).unwrap_err(),
            "condition_budget_exceeded:depth=2:visited_nodes=257:pending_subtrees=1"
        );
    }




    #[test]
    fn negative_and_unrealized_relations_do_not_create_occurrences() {
        let atoms = BTreeMap::from([("m1:a0".into(), atom())]);
        let mut relations = BTreeMap::from([("a".into(), relation("a", "John"))]);
        let graph = EventGraph {
            observations: vec![EventObservation {
                observation_id: "a".into(),
                relation_id: "a".into(),
                native_event_id: None,
                native_event_namespace: None,
                multiplicity: 1,
                quantity_source: None,
            }],
            links: vec![],
        };
        relations.get_mut("a").unwrap().polarity = Polarity::Negative;
        assert_eq!(
            graph.checked_bounds(&relations, &atoms).unwrap_err(),
            "event_not_observed"
        );
        relations.get_mut("a").unwrap().polarity = Polarity::Positive;
        for modality in [
            Modality::Possible,
            Modality::Intended,
            Modality::Required,
            Modality::Counterfactual,
        ] {
            relations.get_mut("a").unwrap().modality = modality;
            assert_eq!(
                graph.checked_bounds(&relations, &atoms).unwrap_err(),
                "event_not_observed"
            );
        }
    }

    #[test]
    fn independent_judgment_cannot_change_claim_sources_or_skip_checks() {
        let atoms = BTreeMap::from([("m1:a0".into(), atom())]);
        let claim = ProposedClaim {
            claim_id: "c1".into(),
            text: "A fact".into(),
            atom_ids: vec!["m1:a0".into()],
            relation_ids: vec![],
            inferred: false,
            count: None,
            condition_binding: None,
        };
        let mut judgment = ClaimJudgment {
            claim_id: "c1".into(),
            verdict: Verdict::Supported,
            atom_ids: claim.atom_ids.clone(),
            roles: Verdict::Supported,
            predicate: Verdict::Supported,
            polarity: Verdict::Supported,
            modality: Verdict::Supported,
            conditions: Verdict::Supported,
            time: Verdict::Supported,
            quantities: Verdict::Supported,
            source_scope: Verdict::Supported,
            reason: "raw evidence checked".into(),
        };
        assert!(checked_claim_atoms(&claim, &judgment, &atoms).is_ok());
        judgment.roles = Verdict::Unknown;
        assert!(checked_claim_atoms(&claim, &judgment, &atoms).is_err());
        judgment.roles = Verdict::Supported;
        judgment.atom_ids.push("other".into());
        assert!(checked_claim_atoms(&claim, &judgment, &atoms).is_err());
    }
}
