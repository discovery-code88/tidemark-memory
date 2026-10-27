use super::*;
use crate::api_types::{AnswerClaim, ExecutionBudget, WorkDispositionCounts};





fn reasoning_sources_unchanged(expected: &[RetrievedItem], current: &[RetrievedItem]) -> bool {
    let snapshots = |items: &[RetrievedItem]| {
        items.iter().map(|item|(item.id,crate::answer_pipeline::reasoning_source_fingerprint(item)))
            .collect::<std::collections::BTreeMap<_,_>>()
    };
    let (left,right)=(snapshots(expected),snapshots(current));
    expected.len()==current.len() && left.len()==expected.len() && right.len()==current.len() && left==right
}

pub(crate) fn apply_verified_reasoning_answer(
    response: &mut AnswerResponse,
    verified: crate::answer_reasoning_provider::VerifiedAnswer,
    sources: &[RetrievedItem],
) {
    let mut claims = Vec::new();
    let mut coverage = Vec::new();
    if let Some(admission) = verified.admission {
        response.semantic_notes.push(SemanticAnswerNote {
            note_type: "reasoning_evidence_admission".into(),
            message: admission.to_string(),
            entity_id: None,
            attribute_key: None,
            scope: None,
            candidate_values: Vec::new(),
        });
    }
    // A revision round can re-propose claims already verified; the delivered
    // answer states each fact once.
    let mut seen_claims = std::collections::HashSet::new();
    for verified_claim in verified
        .claims
        .into_iter()
        .filter(|claim| seen_claims.insert(claim.claim.text.trim().to_lowercase()))
    {
        let ids = verified_claim
            .atoms
            .iter()
            .map(|atom| atom.memory_id)
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let anchors = verified_claim
            .atoms
            .iter()
            .filter_map(|atom| atom.source_anchor.clone())
            .collect::<Vec<_>>();
        let status = if verified_claim.claim.inferred {
            "inferred"
        } else {
            "verified"
        };
        let derivation = vec![
            format!("verification_method:{}", verified.verification.as_str()),
            verified_claim.reason,
        ];
        coverage.push(CoverageReport {
            claim: verified_claim.claim.text.clone(),
            covered: true,
            evidence_ids: ids.clone(),
            evidence_atom_ids: verified_claim.claim.atom_ids.clone(),
            evidence_source_anchors: anchors.clone(),
            derivation: derivation.clone(),
            verification_status: status.into(),
        });
        claims.push(AnswerClaim {
            claim_id: verified_claim.claim.claim_id,
            text: verified_claim.claim.text,
            evidence_ids: ids,
            evidence_atom_ids: verified_claim.claim.atom_ids,
            evidence_source_anchors: anchors,
            derivation,
            support_status: status.into(),
        });
    }
    let used = claims
        .iter()
        .flat_map(|claim| claim.evidence_ids.iter().copied())
        .collect::<std::collections::BTreeSet<_>>();
    // An open list is complete only when its closure is established; the
    // local plan's list scope applies to model-verified entries too.
    let open_list = response.answer_pipeline.as_ref().is_some_and(|receipt| {
        receipt.query_plan.cardinality == "list" && receipt.query_plan.list_scope == "open"
    });
    response.answer_status = if claims.is_empty() {
        "uncertain"
    } else if verified.gaps.is_empty() && !open_list {
        "answered"
    } else {
        "partial"
    }
    .into();
    response.answer = if claims.is_empty() {
        "The available evidence does not establish an answer.".into()
    } else {
        coverage
            .iter()
            .map(|report| crate::memory_engine_summary::checked_claim_cited_text(report, sources))
            .collect::<Vec<_>>()
            .join("\n")
    };
    response.direct_answer =
        crate::memory_engine_summary::direct_answer_display_text(&response.answer, "direct");
    response.answer_mode = if claims.is_empty() {
        "unavailable"
    } else {
        "synthesized"
    }
    .into();
    let quotation = verified.verification == crate::answer_reasoning_provider::VerificationMethod::VerbatimQuotation;
    response.confidence = if quotation { "verbatim_quotation_model_selected" } else { "model_verified_not_deterministic" }.into();
    response.evidence = sources
        .iter()
        .filter(|item| used.contains(&item.id))
        .cloned()
        .collect();
    response.memories = response.evidence.clone();
    response.memory_count = response.evidence.len();
    response.memory_used = used.len();
    response.memory_ignored = sources.len().saturating_sub(used.len());
    response
        .citations
        .retain(|citation| used.contains(&citation.memory_id));
    response.claims = claims;
    response.coverage = coverage;
    if let Some(receipt) = response.answer_pipeline.as_mut() {
        receipt.answer_generator = if quotation { "small_model_selected_verbatim_quotation" } else { "independently_verified_model_claim_plan" }.into();
        receipt.final_evidence_ids = used.iter().copied().collect();
        receipt.claim_plan = response
            .claims
            .iter()
            .map(|claim| crate::api_types::AnswerClaimPlanReceipt {
                claim_id: claim.claim_id.clone(),
                claim_type: if quotation { "verbatim_quotation" } else { "model_verified_relation" }.into(),
                text: Some(claim.text.clone()),
                required_slots: Vec::new(),
                evidence_ids: claim.evidence_ids.clone(),
                evidence_atom_ids: claim.evidence_atom_ids.clone(),
                evidence_source_anchors: claim.evidence_source_anchors.clone(),
                derivation: claim.derivation.clone(),
                support_status: if claim.support_status == "inferred" {
                    "inferred"
                } else {
                    "supported"
                }
                .into(),
            })
            .collect();
        receipt.evidence_plan.selected_answer_atom_ids = response
            .claims
            .iter()
            .flat_map(|claim| claim.evidence_atom_ids.iter().cloned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        receipt.legacy_fallback_used = false;
        receipt.failure_class = if response.claims.is_empty() {
            "claim_verification_failed"
        } else {
            "none"
        }
        .into();
    }
    response.semantic_notes.push(SemanticAnswerNote {
        note_type: "reasoning_coverage".into(),
        message: if quotation { "Coverage gaps from small-model evidence selection and deterministic checks." } else { "Coverage gaps from model planning and independent verification." }.into(),
        entity_id: None,
        attribute_key: None,
        scope: None,
        candidate_values: verified.gaps,
    });
}

pub(crate) async fn canonical_semantic_memory(
    State(state): State<AppState>,
    Json(req): Json<CanonicalSemanticQueryRequest>,
) -> impl IntoResponse {
    let response = {
        let engine = state.engine.read().await;
        engine.query_canonical_semantic(
            req.conversation_id.as_deref(),
            req.entity_id.as_deref(),
            req.attribute_key.as_deref(),
        )
    };

    (StatusCode::OK, Json(response))
}

pub(crate) async fn canonical_semantic_mutation_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<CanonicalSemanticMutationRequest>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&req.conversation_id),
            "canonical semantic mutation",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&req.conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "canonical_semantic_mutation",
                "canonical_semantic",
                &format!("{}:{}", req.entity_id, req.attribute_key),
                Some(tenant_id),
                Some(workspace_id),
                Some(req.conversation_id.clone()),
                "admin scope denied canonical semantic mutation".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let response = {
        let mut engine = state.engine.write().await;
        engine.mutate_canonical_semantic(&req)
    };

    if response.applied {
        if let Err(e) = maybe_persist_memories(state.as_ref(), &response.updated_memory_ids).await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist canonical semantic mutation",
                    "details": { "reason": e, "updated_memory_ids": response.updated_memory_ids }
                })),
            )
                .into_response();
        }
        if let Some(pg) = &state.pg {
            let (conversation_id, semantic_node) = {
                let engine = state.engine.read().await;
                (
                    req.conversation_id.clone(),
                    engine
                        .canonical_semantic_state
                        .get(&response.canonical_key)
                        .cloned(),
                )
            };
            if let Some(node) = semantic_node {
                if let Err(e) = pg
                    .upsert_canonical_semantic_state(
                        &response.canonical_key,
                        &conversation_id,
                        &node,
                    )
                    .await
                {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": "persist_failed",
                            "error_code": "persist_failed",
                            "message": "failed to persist canonical semantic state",
                            "details": { "reason": e.to_string(), "canonical_key": response.canonical_key }
                        })),
                    )
                        .into_response();
                }
            }
            if let Some(audit_entry) = response.audit_entry.as_ref() {
                if let Err(e) = pg.insert_canonical_semantic_audit(audit_entry).await {
                    return (
                        StatusCode::SERVICE_UNAVAILABLE,
                        Json(serde_json::json!({
                            "error": "persist_failed",
                            "error_code": "persist_failed",
                            "message": "failed to persist canonical semantic audit",
                            "details": { "reason": e.to_string(), "canonical_key": response.canonical_key }
                        })),
                    )
                        .into_response();
                }
            }
        }
    }

    if response.applied {
        audit_runtime_write_applied_for_conversation(
            &state,
            auth.as_ref().map(|Extension(a)| a),
            "canonical_semantic_mutation",
            "canonical_semantic",
            &response.canonical_key,
            Some(&req.conversation_id),
            serde_json::json!({
                "action": response.action,
                "updated_memory_ids": response.updated_memory_ids,
                "reason_code": req.reason_code,
                "winner_semantic_id": req.winner_semantic_id,
                "replacement_value": req.replacement_value
            }),
        )
        .await;
    }

    (StatusCode::OK, Json(response)).into_response()
}

pub(crate) async fn activate_procedure_memory(
    State(state): State<AppState>,
    Json(req): Json<ProcedureActivateRequest>,
) -> impl IntoResponse {
    let response = {
        let mut engine = state.engine.write().await;
        engine.activate_procedure(&req.conversation_id, &req.query)
    };

    let Some(response) = response else {
        return api_error(
            StatusCode::NOT_FOUND,
            "procedure_not_found",
            "procedure_not_found",
            "no matching procedure memory found",
            serde_json::json!({
                "conversation_id": req.conversation_id,
                "query": req.query
            }),
        )
        .into_response();
    };

    persist_retrieval_feedback(state.as_ref(), &[response.procedure_id]).await;
    (StatusCode::OK, Json(response)).into_response()
}

fn skill_file_markdown_to_procedure_content(content: &str) -> (String, Vec<String>) {
    let mut trigger_conditions = Vec::new();
    let mut steps = Vec::new();
    let mut rollback_steps = Vec::new();
    let mut success_criteria = Vec::new();
    let mut current_section = "";

    for raw_line in content.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line == "---" {
            continue;
        }
        let heading = line
            .trim_start_matches('#')
            .trim()
            .trim_end_matches(':')
            .to_ascii_lowercase();
        if matches!(
            heading.as_str(),
            "trigger"
                | "triggers"
                | "trigger conditions"
                | "when"
                | "steps"
                | "procedure"
                | "rollback"
                | "rollback steps"
                | "failure rollback"
                | "success"
                | "success criteria"
                | "success criterion"
                | "success standards"
        ) {
            current_section = match heading.as_str() {
                "trigger" | "triggers" | "trigger conditions" | "when" => "trigger",
                "rollback" | "rollback steps" | "failure rollback" => "rollback",
                "success" | "success criteria" | "success criterion" | "success standards" => {
                    "success"
                }
                _ => "steps",
            };
            continue;
        }

        let item = line
            .trim_start_matches(|ch: char| {
                ch == '-' || ch == '*' || ch == '+' || ch.is_ascii_digit() || ch == '.'
            })
            .trim();
        if item.is_empty()
            || item.starts_with("confidence:")
            || item.starts_with("source:")
            || item.starts_with("updated_at:")
        {
            continue;
        }

        if let Some(value) = item
            .strip_prefix("when:")
            .or_else(|| item.strip_prefix("触发条件:"))
            .or_else(|| item.strip_prefix("触发:"))
        {
            trigger_conditions.push(value.trim().to_string());
            continue;
        }
        if let Some(value) = item
            .strip_prefix("step:")
            .or_else(|| item.strip_prefix("步骤:"))
        {
            steps.push(value.trim().to_string());
            continue;
        }
        if let Some(value) = item
            .strip_prefix("rollback:")
            .or_else(|| item.strip_prefix("失败回滚:"))
            .or_else(|| item.strip_prefix("回滚:"))
        {
            rollback_steps.push(value.trim().to_string());
            continue;
        }
        if let Some(value) = item
            .strip_prefix("success:")
            .or_else(|| item.strip_prefix("成功判据:"))
            .or_else(|| item.strip_prefix("成功标准:"))
        {
            success_criteria.push(value.trim().to_string());
            continue;
        }

        match current_section {
            "trigger" => trigger_conditions.push(item.to_string()),
            "rollback" => rollback_steps.push(item.to_string()),
            "success" => success_criteria.push(item.to_string()),
            "steps" => steps.push(item.to_string()),
            _ => {}
        }
    }

    let mut warnings = Vec::new();
    if trigger_conditions.is_empty() {
        warnings.push("missing_trigger_conditions".to_string());
    }
    if steps.is_empty() {
        warnings.push("missing_steps".to_string());
    }
    if rollback_steps.is_empty() {
        warnings.push("missing_rollback_steps".to_string());
    }
    if success_criteria.is_empty() {
        warnings.push("missing_success_criteria".to_string());
    }

    let mut lines = Vec::new();
    lines.extend(
        trigger_conditions
            .iter()
            .map(|item| format!("when: {item}")),
    );
    lines.extend(steps.iter().map(|item| format!("step: {item}")));
    lines.extend(
        rollback_steps
            .iter()
            .map(|item| format!("rollback: {item}")),
    );
    lines.extend(
        success_criteria
            .iter()
            .map(|item| format!("success: {item}")),
    );
    if lines.is_empty() {
        lines.push(format!("step: {}", compact_context_line(content, 500)));
    }
    (lines.join("\n"), warnings)
}

pub(crate) async fn import_procedure_skill_file_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<ProcedureImportSkillFileRequest>,
) -> Response {
    let space_id = req
        .space_id
        .clone()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "default".to_string());
    if !req.path.trim().starts_with("skills/") || !req.path.trim().ends_with(".md") {
        return api_error(
            StatusCode::BAD_REQUEST,
            "invalid_skill_file",
            "invalid_skill_file_path",
            "procedure skill import requires a Memory File path under skills/*.md",
            serde_json::json!({ "path": req.path }),
        )
        .into_response();
    }
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&space_id),
            "procedure skill file import",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&space_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "procedure_import_skill_file",
                "procedure",
                &req.path,
                Some(tenant_id),
                Some(workspace_id),
                Some(space_id.clone()),
                "admin scope denied procedure skill file import".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let (safe_path, content) = match read_memory_file_content_for_space(&space_id, &req.path).await
    {
        Ok(value) => value,
        Err(response) => return response,
    };
    let (procedure_content, validation_warnings) =
        skill_file_markdown_to_procedure_content(&content);
    let entity = safe_path
        .trim_start_matches("skills/")
        .trim_end_matches(".md")
        .replace('/', "::");
    let ingest_req = IngestRequest {
        conversation_id: space_id.clone(),
        layer: MemoryLayer::Procedure,
        content: procedure_content,
        source: Some(
            req.source
                .clone()
                .unwrap_or_else(|| format!("skill_file_import:{safe_path}")),
        ),
        trust: Some(req.trust.unwrap_or(0.82).clamp(0.0, 1.0)),
        entity: Some(format!("skill:{entity}")),
        fact_key: Some("procedure".to_string()),
        fact_value: Some(safe_path.clone()),
    };
    let (ingest, created_ids) = {
        let mut engine = state.engine.write().await;
        let before_len = engine.memories.len();
        let ingest = engine.ingest(ingest_req);
        let created_ids = engine
            .memories
            .iter()
            .skip(before_len)
            .map(|memory| memory.id)
            .collect::<Vec<_>>();
        (ingest, created_ids)
    };
    if !created_ids.is_empty() {
        if let Err(error) = maybe_persist_memories(state.as_ref(), &created_ids).await {
            let rolled_back = rollback_memories(state.as_ref(), &created_ids).await;
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist imported skill procedure",
                    "details": {
                        "reason": error,
                        "rolled_back": rolled_back,
                        "path": safe_path
                    }
                })),
            )
                .into_response();
        }
    }

    let memory_id = ingest.memory_id;
    let procedure_schema = if let Some(memory_id) = memory_id {
        let engine = state.engine.read().await;
        engine
            .memories
            .iter()
            .find(|memory| memory.id == memory_id)
            .and_then(|memory| memory.procedure_schema.clone())
    } else {
        None
    };
    let activation = if let Some(query) = req.activate_query.as_deref() {
        let mut engine = state.engine.write().await;
        engine.activate_procedure(&space_id, query)
    } else {
        None
    };
    if let Some(activation) = &activation {
        persist_retrieval_feedback(state.as_ref(), &[activation.procedure_id]).await;
    }
    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "procedure_import_skill_file",
        "procedure",
        &safe_path,
        Some(&space_id),
        serde_json::json!({
            "memory_id": memory_id,
            "validation_warnings": validation_warnings.clone(),
            "created_ids": created_ids
        }),
    )
    .await;

    let accepted = ingest.accepted;
    (
        StatusCode::OK,
        Json(ProcedureImportSkillFileResponse {
            import_format: "procedure_skill_file_import_v1".to_string(),
            generated_at: Utc::now(),
            space_id: space_id.clone(),
            path: safe_path.clone(),
            accepted,
            memory_id,
            procedure_schema,
            validation_warnings,
            activation,
            ingest,
            rollback_endpoint: "/memory/procedure/rollback-import".to_string(),
            rollback_payload: serde_json::json!({
                "conversation_id": space_id,
                "procedure_id": memory_id,
                "path": safe_path,
                "reason": "rollback imported skill procedure"
            }),
            trace_group: "procedure".to_string(),
            message: if accepted {
                "Imported skill file as a typed procedure memory.".to_string()
            } else {
                "Skill file import was rejected by memory write policy.".to_string()
            },
        }),
    )
        .into_response()
}

pub(crate) async fn rollback_imported_procedure_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<ProcedureImportRollbackRequest>,
) -> Response {
    let space_id = req
        .space_id
        .clone()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "default".to_string());
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&space_id),
            "procedure skill import rollback",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&space_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "procedure_import_rollback",
                "procedure",
                &req.procedure_id.to_string(),
                Some(tenant_id),
                Some(workspace_id),
                Some(space_id.clone()),
                "admin scope denied procedure import rollback".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let rollback_allowed = {
        let engine = state.engine.read().await;
        engine.memories.iter().any(|memory| {
            memory.id == req.procedure_id
                && memory.conversation_id == space_id
                && memory.layer == MemoryLayer::Procedure
                && req
                    .path
                    .as_deref()
                    .map(|path| {
                        memory.source == format!("skill_file_import:{path}")
                            || memory.fact_value.as_deref() == Some(path)
                    })
                    .unwrap_or_else(|| memory.source.starts_with("skill_file_import:"))
        })
    };
    if !rollback_allowed {
        return api_error(
            StatusCode::NOT_FOUND,
            "procedure_rollback_not_found",
            "procedure_import_not_found",
            "imported skill procedure was not found or does not match the rollback request",
            serde_json::json!({
                "conversation_id": space_id,
                "procedure_id": req.procedure_id,
                "path": req.path
            }),
        )
        .into_response();
    }

    let rolled_back = rollback_memory(state.as_ref(), req.procedure_id).await;
    let activation_after_rollback = {
        let mut engine = state.engine.write().await;
        let query = req
            .path
            .as_deref()
            .unwrap_or("imported skill procedure rollback");
        engine.activate_procedure(&space_id, query)
    };
    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "procedure_import_rollback",
        "procedure",
        &req.procedure_id.to_string(),
        Some(&space_id),
        serde_json::json!({
            "rolled_back": rolled_back,
            "path": req.path,
            "actor": req.actor
        }),
    )
    .await;

    let reason = req
        .reason
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("rollback imported skill procedure")
        .to_string();
    (
        StatusCode::OK,
        Json(ProcedureImportRollbackResponse {
            rollback_format: "procedure_skill_import_rollback_v1".to_string(),
            generated_at: Utc::now(),
            space_id,
            procedure_id: req.procedure_id,
            path: req.path,
            rolled_back,
            reason,
            activation_after_rollback,
            message: if rolled_back {
                "Rolled back imported skill procedure; future activation will not use this procedure memory.".to_string()
            } else {
                "No imported skill procedure was removed.".to_string()
            },
        }),
    )
        .into_response()
}

pub(crate) async fn sync_procedure_skill_files_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<ProcedureSyncSkillFilesRequest>,
) -> Response {
    let space_id = req
        .space_id
        .clone()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "default".to_string());
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&space_id),
            "procedure skill file sync",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&space_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "procedure_sync_skill_files",
                "procedure",
                "skills/*.md",
                Some(tenant_id),
                Some(workspace_id),
                Some(space_id.clone()),
                "admin scope denied procedure skill file sync".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let force = req.force.unwrap_or(false);
    let records = collect_memory_file_records_for_space(&space_id, true).await;
    let mut items = Vec::new();
    let mut imported_count = 0usize;
    let mut updated_count = 0usize;
    let mut unchanged_count = 0usize;
    let mut failed_count = 0usize;

    for record in records
        .into_iter()
        .filter(|record| record.path.starts_with("skills/") && record.path.ends_with(".md"))
        .filter(|record| record.path != "skills/README.md")
    {
        let path = record.path.clone();
        let content = record.content.unwrap_or_default();
        let (procedure_content, validation_warnings) =
            skill_file_markdown_to_procedure_content(&content);
        let previous = {
            let engine = state.engine.read().await;
            engine
                .memories
                .iter()
                .filter(|memory| {
                    memory.conversation_id == space_id
                        && memory.layer == MemoryLayer::Procedure
                        && (memory.source == format!("skill_file_import:{path}")
                            || memory.fact_value.as_deref() == Some(path.as_str()))
                })
                .max_by_key(|memory| memory.created_at)
                .map(|memory| (memory.id, memory.content.clone()))
        };
        if let Some((previous_id, previous_content)) = previous.clone() {
            if !force && previous_content == procedure_content {
                unchanged_count += 1;
                items.push(ProcedureSyncSkillFileItem {
                    path,
                    status: "unchanged".to_string(),
                    previous_procedure_id: Some(previous_id),
                    memory_id: Some(previous_id),
                    validation_warnings,
                    rollback_endpoint: Some("/memory/procedure/rollback-import".to_string()),
                    rollback_payload: Some(serde_json::json!({
                        "conversation_id": space_id.clone(),
                        "procedure_id": previous_id,
                        "path": record.path,
                        "reason": "rollback synced skill procedure"
                    })),
                });
                continue;
            }
            let _ = rollback_memory(state.as_ref(), previous_id).await;
        }

        let entity = path
            .trim_start_matches("skills/")
            .trim_end_matches(".md")
            .replace('/', "::");
        let ingest_req = IngestRequest {
            conversation_id: space_id.clone(),
            layer: MemoryLayer::Procedure,
            content: procedure_content,
            source: Some(format!("skill_file_import:{path}")),
            trust: Some(req.trust.unwrap_or(0.82).clamp(0.0, 1.0)),
            entity: Some(format!("skill:{entity}")),
            fact_key: Some("procedure".to_string()),
            fact_value: Some(path.clone()),
        };
        let (ingest, created_ids) = {
            let mut engine = state.engine.write().await;
            let before_len = engine.memories.len();
            let ingest = engine.ingest(ingest_req);
            let created_ids = engine
                .memories
                .iter()
                .skip(before_len)
                .map(|memory| memory.id)
                .collect::<Vec<_>>();
            (ingest, created_ids)
        };
        if !created_ids.is_empty() {
            if let Err(error) = maybe_persist_memories(state.as_ref(), &created_ids).await {
                let _ = rollback_memories(state.as_ref(), &created_ids).await;
                failed_count += 1;
                items.push(ProcedureSyncSkillFileItem {
                    path,
                    status: format!("failed_persist: {error}"),
                    previous_procedure_id: previous.map(|(id, _)| id),
                    memory_id: None,
                    validation_warnings,
                    rollback_endpoint: None,
                    rollback_payload: None,
                });
                continue;
            }
        }
        let memory_id = ingest.memory_id;
        if previous.is_some() {
            updated_count += 1;
        } else {
            imported_count += 1;
        }
        items.push(ProcedureSyncSkillFileItem {
            path: path.clone(),
            status: if previous.is_some() {
                "updated".to_string()
            } else {
                "imported".to_string()
            },
            previous_procedure_id: previous.map(|(id, _)| id),
            memory_id,
            validation_warnings,
            rollback_endpoint: Some("/memory/procedure/rollback-import".to_string()),
            rollback_payload: Some(serde_json::json!({
                "conversation_id": space_id.clone(),
                "procedure_id": memory_id,
                "path": path,
                "reason": "rollback synced skill procedure"
            })),
        });
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "procedure_sync_skill_files",
        "procedure",
        "skills/*.md",
        Some(&space_id),
        serde_json::json!({
            "imported_count": imported_count,
            "updated_count": updated_count,
            "unchanged_count": unchanged_count,
            "failed_count": failed_count,
        }),
    )
    .await;

    (
        StatusCode::OK,
        Json(ProcedureSyncSkillFilesResponse {
            sync_format: "procedure_skill_file_sync_v1".to_string(),
            generated_at: Utc::now(),
            space_id,
            scanned_count: items.len(),
            imported_count,
            updated_count,
            unchanged_count,
            failed_count,
            items,
        }),
    )
        .into_response()
}

pub(crate) async fn execute_procedure_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<ProcedureExecutionRequest>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&req.conversation_id),
            "procedure execute",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&req.conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "procedure_execute",
                "procedure",
                &req.procedure_id.to_string(),
                Some(tenant_id),
                Some(workspace_id),
                Some(req.conversation_id.clone()),
                "admin scope denied procedure execute".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let execution = {
        let mut engine = state.engine.write().await;
        engine.execute_procedure(&req)
    };

    let Some(execution) = execution else {
        return api_error(
            StatusCode::NOT_FOUND,
            "procedure_execution_failed",
            "procedure_execution_failed",
            "matching procedure could not be executed",
            serde_json::json!({
                "conversation_id": req.conversation_id,
                "procedure_id": req.procedure_id
            }),
        )
        .into_response();
    };

    if let Err(e) = maybe_persist_memories(state.as_ref(), &execution.created_ids).await {
        let rolled_back = rollback_memories(state.as_ref(), &execution.created_ids).await;
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "persist_failed",
                "error_code": "persist_failed",
                "message": "failed to persist procedure execution memories",
                "details": {
                    "reason": e,
                    "rolled_back": rolled_back,
                    "procedure_id": req.procedure_id
                }
            })),
        )
            .into_response();
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "procedure_execute",
        "procedure",
        &req.procedure_id.to_string(),
        Some(&req.conversation_id),
        serde_json::json!({
            "execution_status": execution.response.execution_status,
            "execution_memory_id": execution.response.execution_memory_id,
            "event_id": execution.response.event_id,
            "created_ids": execution.created_ids,
            "failure_reason": execution.response.failure_reason,
            "success_evidence": execution.response.success_evidence
        }),
    )
    .await;

    (StatusCode::OK, Json(execution.response)).into_response()
}

pub(crate) async fn procedure_status_memory(
    State(state): State<AppState>,
    Json(req): Json<ProcedureStatusRequest>,
) -> impl IntoResponse {
    let state_response = {
        let engine = state.engine.read().await;
        engine.latest_procedure_execution_state(&req.conversation_id, req.procedure_id)
    };

    let Some(state_response) = state_response else {
        return api_error(
            StatusCode::NOT_FOUND,
            "procedure_status_not_found",
            "procedure_status_not_found",
            "no procedure execution state found",
            serde_json::json!({
                "conversation_id": req.conversation_id,
                "procedure_id": req.procedure_id
            }),
        )
        .into_response();
    };

    (
        StatusCode::OK,
        Json(ProcedureStatusResponse {
            procedure_id: state_response.procedure_id,
            current_step_index: state_response.current_step_index,
            retry_count: state_response.retry_count,
            outcome: state_response.outcome,
            failure_reason: state_response.failure_reason,
            success_evidence: state_response.success_evidence,
            steps: state_response.steps,
        }),
    )
        .into_response()
}

pub(crate) async fn consolidate_procedure_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<ProcedureConsolidateRequest>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&req.conversation_id),
            "procedure consolidate",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&req.conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "procedure_consolidate",
                "procedure",
                &req.procedure_id.to_string(),
                Some(tenant_id),
                Some(workspace_id),
                Some(req.conversation_id.clone()),
                "admin scope denied procedure consolidate".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let response = {
        let mut engine = state.engine.write().await;
        engine.consolidate_procedure(&req.conversation_id, req.procedure_id)
    };

    let Some(response) = response else {
        return api_error(
            StatusCode::NOT_FOUND,
            "procedure_consolidation_failed",
            "procedure_consolidation_failed",
            "matching procedure could not be consolidated",
            serde_json::json!({
                "conversation_id": req.conversation_id,
                "procedure_id": req.procedure_id
            }),
        )
        .into_response();
    };

    if let Some(procedure_id) = response.procedure_id {
        if let Err(e) = maybe_persist_memory(state.as_ref(), procedure_id).await {
            let rolled_back = rollback_memory(state.as_ref(), procedure_id).await;
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist consolidated procedure",
                    "details": {
                        "reason": e,
                        "rolled_back": rolled_back,
                        "procedure_id": procedure_id
                    }
                })),
            )
                .into_response();
        }
    }

    if !response.synthesized_fact_ids.is_empty() {
        if let Err(e) = maybe_persist_memories(state.as_ref(), &response.synthesized_fact_ids).await
        {
            let rolled_back =
                rollback_memories(state.as_ref(), &response.synthesized_fact_ids).await;
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist synthesized semantic facts",
                    "details": {
                        "reason": e,
                        "rolled_back": rolled_back,
                        "procedure_id": response.procedure_id,
                        "synthesized_fact_ids": response.synthesized_fact_ids
                    }
                })),
            )
                .into_response();
        }
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "procedure_consolidate",
        "procedure",
        &req.procedure_id.to_string(),
        Some(&req.conversation_id),
        serde_json::json!({
            "consolidated": response.consolidated,
            "procedure_id": response.procedure_id,
            "version": response.version,
            "synthesized_fact_ids": response.synthesized_fact_ids,
            "synthesized_failure_motifs": response.synthesized_failure_motifs
        }),
    )
    .await;

    (StatusCode::OK, Json(response)).into_response()
}

pub(crate) async fn evaluate_procedure_memory(
    State(state): State<AppState>,
    Json(req): Json<ProcedureEvaluateRequest>,
) -> impl IntoResponse {
    let response = {
        let engine = state.engine.read().await;
        engine.evaluate_procedure(&req.conversation_id, req.procedure_id)
    };

    let Some(response) = response else {
        return api_error(
            StatusCode::NOT_FOUND,
            "procedure_evaluation_failed",
            "procedure_evaluation_failed",
            "matching procedure could not be evaluated",
            serde_json::json!({
                "conversation_id": req.conversation_id,
                "procedure_id": req.procedure_id
            }),
        )
        .into_response();
    };

    (StatusCode::OK, Json(response)).into_response()
}

pub(crate) async fn summarize_memory(
    Path(conversation_id): Path<String>,
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&conversation_id),
            "summarize",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "summarize",
                "conversation",
                &conversation_id,
                Some(tenant_id),
                Some(workspace_id),
                Some(conversation_id.clone()),
                "admin scope denied summarize".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let resp = {
        let mut engine = state.engine.write().await;
        engine.summarize_conversation(&conversation_id)
    };

    if let Some(summary_id) = resp.summary_id {
        if let Err(e) = maybe_persist_memory(state.as_ref(), summary_id).await {
            let rolled_back = rollback_memory(state.as_ref(), summary_id).await;
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist summary memory",
                    "details": {
                        "reason": e,
                        "persistence_error": e,
                        "rolled_back": rolled_back,
                        "summary_id": summary_id,
                        "summary": resp.summary,
                        "anchors": resp.anchors
                    }
                })),
            )
                .into_response();
        }
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "summarize",
        "conversation",
        &conversation_id,
        Some(&conversation_id),
        serde_json::json!({
            "summary_id": resp.summary_id,
            "anchor_count": resp.anchors.len()
        }),
    )
    .await;

    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap_or_default()),
    )
        .into_response()
}

pub(crate) async fn consolidate_memory(
    Path(conversation_id): Path<String>,
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&conversation_id),
            "consolidate",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "consolidate",
                "conversation",
                &conversation_id,
                Some(tenant_id),
                Some(workspace_id),
                Some(conversation_id.clone()),
                "admin scope denied consolidate".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let resp = {
        let mut engine = state.engine.write().await;
        engine.consolidate_conversation(&conversation_id)
    };

    if let Some(summary_id) = resp.summary_id {
        if let Err(e) = maybe_persist_memory(state.as_ref(), summary_id).await {
            let rolled_back = rollback_memory(state.as_ref(), summary_id).await;
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist consolidation summary",
                    "details": {
                        "reason": e,
                        "persistence_error": e,
                        "rolled_back": rolled_back,
                        "summary_id": summary_id,
                        "summary": resp.summary,
                        "anchors": resp.anchors
                    }
                })),
            )
                .into_response();
        }
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "consolidate",
        "conversation",
        &conversation_id,
        Some(&conversation_id),
        serde_json::json!({
            "summary_id": resp.summary_id,
            "anchor_count": resp.anchors.len()
        }),
    )
    .await;

    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap_or_default()),
    )
        .into_response()
}

pub(crate) async fn prune_memory(
    Path(conversation_id): Path<String>,
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&conversation_id),
            "prune",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "prune",
                "conversation",
                &conversation_id,
                Some(tenant_id),
                Some(workspace_id),
                Some(conversation_id.clone()),
                "admin scope denied prune".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let resp = {
        let mut engine = state.engine.write().await;
        engine.prune_ignored_memories(&conversation_id)
    };

    if let Some(pg) = &state.pg {
        let _ = pg.delete_memories(&resp.pruned_ids).await;
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "prune",
        "conversation",
        &conversation_id,
        Some(&conversation_id),
        serde_json::json!({
            "pruned": resp.pruned,
            "pruned_ids": resp.pruned_ids
        }),
    )
    .await;

    (StatusCode::OK, Json(resp)).into_response()
}

pub(crate) async fn decay_memory(
    Path(conversation_id): Path<String>,
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&conversation_id),
            "decay",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "decay",
                "conversation",
                &conversation_id,
                Some(tenant_id),
                Some(workspace_id),
                Some(conversation_id.clone()),
                "admin scope denied decay".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let resp = {
        let mut engine = state.engine.write().await;
        engine.decay_stale_memories(&conversation_id)
    };

    if let Some(pg) = &state.pg {
        let _ = pg.delete_memories(&resp.decayed_ids).await;
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "decay",
        "conversation",
        &conversation_id,
        Some(&conversation_id),
        serde_json::json!({
            "decayed": resp.decayed,
            "decayed_ids": resp.decayed_ids
        }),
    )
    .await;

    (StatusCode::OK, Json(resp)).into_response()
}

pub(crate) async fn semantic_sleep_cycle_memory(
    Path(conversation_id): Path<String>,
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
) -> impl IntoResponse {
    if let Some(response) = {
        let engine = state.engine.read().await;
        let response = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            Some(&conversation_id),
            "semantic sleep cycle",
        );
        if response.is_some() {
            let workspace_id = engine.workspace_for_conversation(&conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "semantic_sleep_cycle",
                "conversation",
                &conversation_id,
                Some(tenant_id),
                Some(workspace_id),
                Some(conversation_id.clone()),
                "admin scope denied semantic sleep cycle".to_string(),
                "admin_scope_forbidden",
            )
            .await;
        }
        response
    } {
        return response;
    }

    let mut resp = {
        let mut engine = state.engine.write().await;
        engine.run_semantic_sleep_cycle(&conversation_id)
    };


    if let Some(pg) = &state.pg {
        let (memory_ids, audit_entries, canonical_nodes) = {
            let engine = state.engine.read().await;
            let memory_ids = engine
                .memories
                .iter()
                .filter(|memory| memory.conversation_id == conversation_id)
                .filter_map(|memory| {
                    memory.semantic_node.as_ref().and_then(|node| {
                        (engine.canonical_semantic_state.contains_key(
                            &MemoryEngine::canonical_semantic_key(
                                &node.entity_id,
                                &node.attribute_key,
                            ),
                        ))
                        .then_some(memory.id)
                    })
                })
                .collect::<Vec<_>>();
            let audit_entries = engine
                .canonical_semantic_audit
                .iter()
                .filter(|entry| entry.conversation_id == conversation_id)
                .cloned()
                .collect::<Vec<_>>();
            let canonical_nodes = engine
                .canonical_semantic_state
                .iter()
                .filter(|(_, node)| {
                    engine.memories.iter().any(|memory| {
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
            (memory_ids, audit_entries, canonical_nodes)
        };

        if let Err(e) = maybe_persist_memories(state.as_ref(), &memory_ids).await {
            return (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(serde_json::json!({
                    "error": "persist_failed",
                    "error_code": "persist_failed",
                    "message": "failed to persist sleep-cycle updates",
                    "details": { "reason": e }
                })),
            )
                .into_response();
        }
        for entry in audit_entries {
            let _ = pg.insert_canonical_semantic_audit(&entry).await;
        }
        for (key, node) in canonical_nodes {
            let _ = pg
                .upsert_canonical_semantic_state(&key, &conversation_id, &node)
                .await;
        }
    }

    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "semantic_sleep_cycle",
        "conversation",
        &conversation_id,
        Some(&conversation_id),
        serde_json::json!({
            "job_id": resp.job_id,
            "job_status": resp.job_status,
            "repaired_nodes": resp.repaired_nodes,
            "resolved_conflicts": resp.resolved_conflicts,
            "deprecated_nodes": resp.deprecated_nodes
        }),
    )
    .await;

    (StatusCode::OK, Json(resp)).into_response()
}

pub(crate) fn spawn_semantic_sleep_worker(state: AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(state.sleep_cycle_interval);
        loop {
            interval.tick().await;
            if state.shutting_down.load(AtomicOrdering::Relaxed) {
                break;
            }

            let mut queued = {
                let engine = state.engine.read().await;
                engine
                    .memories
                    .iter()
                    .map(|memory| memory.conversation_id.clone())
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .map(|conversation_id| {
                        let (priority, trigger_reason) =
                            engine.consolidation_priority_for_conversation(&conversation_id);
                        (conversation_id, priority, trigger_reason)
                    })
                    .collect::<Vec<_>>()
            };
            queued.sort_by(|a, b| b.1.cmp(&a.1));

            for (conversation_id, priority, trigger_reason) in queued {
                if state.shutting_down.load(AtomicOrdering::Relaxed) {
                    break;
                }
                if priority < 15 {
                    continue;
                }
                let report = {
                    let mut engine = state.engine.write().await;
                    if let Some(job) = engine.consolidation_jobs.iter_mut().find(|job| {
                        job.conversation_id.as_deref() == Some(conversation_id.as_str())
                            && job.status == "running"
                    }) {
                        job.trigger_reason = trigger_reason.clone();
                        job.priority = priority;
                    }
                    engine.run_semantic_sleep_cycle(&conversation_id)
                };
                if report.repaired_nodes == 0 && report.deprecated_nodes == 0 {
                    continue;
                }

                if state.pg.is_some() {
                    let (memory_ids, audit_entries, canonical_nodes) = {
                        let engine = state.engine.read().await;
                        let memory_ids = engine
                            .memories
                            .iter()
                            .filter(|memory| memory.conversation_id == conversation_id)
                            .filter_map(|memory| {
                                memory.semantic_node.as_ref().and_then(|node| {
                                    (engine.canonical_semantic_state.contains_key(
                                        &MemoryEngine::canonical_semantic_key(
                                            &node.entity_id,
                                            &node.attribute_key,
                                        ),
                                    ))
                                    .then_some(memory.id)
                                })
                            })
                            .collect::<Vec<_>>();
                        let audit_entries = engine
                            .canonical_semantic_audit
                            .iter()
                            .filter(|entry| entry.conversation_id == conversation_id)
                            .cloned()
                            .collect::<Vec<_>>();
                        let canonical_nodes = engine
                            .canonical_semantic_state
                            .iter()
                            .filter(|(_, node)| {
                                engine.memories.iter().any(|memory| {
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
                        (memory_ids, audit_entries, canonical_nodes)
                    };

                    let _ = maybe_persist_memories(state.as_ref(), &memory_ids).await;
                    if let Some(pg) = &state.pg {
                        for entry in audit_entries {
                            let _ = pg.insert_canonical_semantic_audit(&entry).await;
                        }
                        for (key, node) in canonical_nodes {
                            let _ = pg
                                .upsert_canonical_semantic_state(&key, &conversation_id, &node)
                                .await;
                        }
                        let jobs = {
                            let engine = state.engine.read().await;
                            engine
                                .consolidation_jobs
                                .iter()
                                .filter(|job| {
                                    job.conversation_id.as_deref() == Some(conversation_id.as_str())
                                })
                                .cloned()
                                .collect::<Vec<_>>()
                        };
                        for job in jobs {
                            let _ = pg.upsert_consolidation_job(&job).await;
                        }
                    }
                }
            }
        }
    });
}

pub(crate) async fn answer_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<AnswerRequest>,
) -> Response {
    let timeout_ms = crate::answer_execution::answer_deadline_ms(req.reasoning_mode.unwrap_or_default());
    answer_memory_with_execution(
        state,
        auth,
        req,
        ExecutionBudget {
            timeout_ms: Some(timeout_ms),
            ..ExecutionBudget::default()
        },
        "auto".to_string(),
    )
    .await
}

pub(crate) async fn answer_memory_from_intent_workspace(
    state: AppState,
    auth: Option<Extension<AuthContext>>,
    req: AnswerRequest,
) -> Response {
    let timeout_ms = crate::answer_execution::answer_deadline_ms(req.reasoning_mode.unwrap_or_default());
    answer_memory_with_execution(
        state,
        auth,
        req,
        ExecutionBudget {
            timeout_ms: Some(timeout_ms),
            ..ExecutionBudget::default()
        },
        "intent_workspace_selection".to_string(),
    )
    .await
}

const MAX_EXPLICIT_ANSWER_SELECTION: usize = 512;

fn validate_answer_shape(req: &AnswerRequest) -> Result<(), Response> {
    if req
        .response_mode
        .as_deref()
        .is_some_and(|value| !crate::memory_engine_summary::answer_response_mode_is_valid(value))
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_shape_error",
            "invalid_response_mode",
            "response_mode must be auto, direct, evidence, or verbatim",
            serde_json::json!({
                "allowed": ["auto", "direct", "evidence", "verbatim"]
            }),
        )
        .into_response());
    }
    if req
        .answer_nodes
        .is_some_and(|value| value == 0 || value > 64)
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_shape_error",
            "invalid_answer_nodes",
            "answer_nodes must be between 1 and 64",
            serde_json::json!({"maximum": 64}),
        )
        .into_response());
    }
    if req
        .source_limit
        .is_some_and(|value| value == 0 || value > 512)
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_shape_error",
            "invalid_source_limit",
            "source_limit must be between 1 and 512",
            serde_json::json!({"maximum": 512}),
        )
        .into_response());
    }
    if req
        .document_limit
        .is_some_and(|value| value == 0 || value > 50)
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_shape_error",
            "invalid_document_limit",
            "document_limit must be between 1 and 50",
            serde_json::json!({"maximum": 50}),
        )
        .into_response());
    }
    if let Some(detail_level) = req.detail_level.as_deref() {
        let detail_level = detail_level.trim().to_ascii_lowercase();
        if !matches!(
            detail_level.as_str(),
            "auto" | "concise" | "balanced" | "detailed" | "custom"
        ) {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "answer_shape_error",
                "invalid_detail_level",
                "detail_level must be auto, concise, balanced, detailed, or custom",
                serde_json::json!({}),
            )
            .into_response());
        }
    }
    Ok(())
}

fn validate_explicit_answer_selection(
    engine: &MemoryEngine,
    req: &AnswerRequest,
) -> Result<(), Response> {
    let Some(selected_memory_ids) = req.selected_memory_ids.as_ref() else {
        if req.selection_binding.is_some() {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "answer_selection_error",
                "selection_binding_without_memory_scope",
                "an internal selection binding requires selected_memory_ids",
                serde_json::json!({}),
            )
            .into_response());
        }
        return Ok(());
    };
    let Some(space_id) = req
        .conversation_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_selection_error",
            "selection_space_required",
            "conversation_id is required when selected_memory_ids is present",
            serde_json::json!({}),
        )
        .into_response());
    };
    if selected_memory_ids.is_empty()
        || selected_memory_ids.len() > MAX_EXPLICIT_ANSWER_SELECTION
        || selected_memory_ids.iter().any(|memory_id| *memory_id == 0)
    {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_selection_error",
            "invalid_selected_memory_ids",
            format!(
                "selected_memory_ids must contain 1-{MAX_EXPLICIT_ANSWER_SELECTION} positive IDs"
            ),
            serde_json::json!({
                "selected_count": selected_memory_ids.len(),
                "maximum": MAX_EXPLICIT_ANSWER_SELECTION,
            }),
        )
        .into_response());
    }
    let unique = selected_memory_ids.iter().copied().collect::<HashSet<_>>();
    if unique.len() != selected_memory_ids.len() {
        return Err(api_error(
            StatusCode::BAD_REQUEST,
            "answer_selection_error",
            "duplicate_selected_memory_ids",
            "selected_memory_ids must not contain duplicates",
            serde_json::json!({}),
        )
        .into_response());
    }

    let mut missing_ids = Vec::new();
    let mut outside_space_ids = Vec::new();
    let mut ineligible_ids = Vec::new();
    for memory_id in selected_memory_ids {
        let Some(memory) = engine
            .memories
            .iter()
            .find(|memory| memory.id == *memory_id)
        else {
            missing_ids.push(*memory_id);
            continue;
        };
        if memory.conversation_id != space_id {
            outside_space_ids.push(*memory_id);
        } else if !crate::memory_engine_summary::memory_record_is_eligible_for_selected_answer(
            memory,
        ) {
            ineligible_ids.push(*memory_id);
        }
    }
    if !outside_space_ids.is_empty() {
        return Err(api_error(
            StatusCode::FORBIDDEN,
            "answer_selection_error",
            "selected_memory_outside_space",
            "selected memories must belong to the requested conversation",
            serde_json::json!({"rejected_memory_ids": outside_space_ids}),
        )
        .into_response());
    }
    if !missing_ids.is_empty() {
        return Err(api_error(
            StatusCode::NOT_FOUND,
            "answer_selection_error",
            "selected_memory_not_found",
            "one or more selected memories do not exist",
            serde_json::json!({"missing_memory_ids": missing_ids}),
        )
        .into_response());
    }
    if !ineligible_ids.is_empty() {
        return Err(api_error(
            StatusCode::CONFLICT,
            "answer_selection_error",
            "selected_memory_not_retrievable",
            "selected memories are stale, superseded, deprecated, ignored, or inactive",
            serde_json::json!({"rejected_memory_ids": ineligible_ids}),
        )
        .into_response());
    }
    if let Some(binding) = req.selection_binding.as_ref() {
        let bound_ids = binding
            .memory_bindings
            .iter()
            .map(|item| item.memory_id)
            .collect::<Vec<_>>();
        if bound_ids.as_slice() != selected_memory_ids.as_slice() {
            return Err(api_error(
                StatusCode::CONFLICT,
                "answer_selection_error",
                "selection_binding_mismatch",
                "the workspace selection no longer matches its bound memory scope",
                serde_json::json!({}),
            )
            .into_response());
        }
        let stale_ids = binding
            .memory_bindings
            .iter()
            .filter_map(|bound| {
                engine
                    .memories
                    .iter()
                    .find(|memory| memory.id == bound.memory_id)
                    .filter(|memory| {
                        memory.version != bound.memory_version
                            || crate::memory_engine_summary::selected_answer_source_record_digest(
                                memory,
                            ) != bound.source_record_digest
                    })
                    .map(|_| bound.memory_id)
            })
            .collect::<Vec<_>>();
        if !stale_ids.is_empty() {
            return Err(api_error(
                StatusCode::CONFLICT,
                "answer_selection_error",
                "workspace_selection_source_stale",
                "selected workspace sources changed after synchronization; synchronize sources before answering",
                serde_json::json!({"stale_memory_ids": stale_ids}),
            )
            .into_response());
        }
    }
    Ok(())
}

pub(crate) async fn answer_memory_http(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    headers: HeaderMap,
    Json(req): Json<AnswerRequest>,
) -> Response {
    let (budget, workload_shape) = execution_budget_from_headers(
        &headers,
        crate::answer_execution::answer_deadline_ms(req.reasoning_mode.unwrap_or_default()),
    );
    answer_memory_with_execution(state, auth, req, budget, workload_shape).await
}

async fn answer_memory_with_execution(
    state: AppState,
    auth: Option<Extension<AuthContext>>,
    req: AnswerRequest,
    budget: ExecutionBudget,
    workload_shape: String,
) -> Response {
    let execution = Arc::new(crate::answer_execution::AnswerExecution::new(
        req.reasoning_mode.unwrap_or_default(),
        &budget,
    ));
    answer_memory_with_control(state, auth, req, budget, workload_shape, execution).await
}

async fn answer_memory_with_control(
    state: AppState,
    auth: Option<Extension<AuthContext>>,
    req: AnswerRequest,
    mut budget: ExecutionBudget,
    workload_shape: String,
    execution: Arc<crate::answer_execution::AnswerExecution>,
) -> Response {
    budget.timeout_ms = Some(
        execution
            .deadline
            .duration_since(execution.started)
            .as_millis() as u64,
    );
    if tokio::time::Instant::now() < execution.deadline {
        if let Ok(response) = tokio::time::timeout_at(
            execution.deadline,
            answer_memory_until(
                state.clone(),
                auth,
                req.clone(),
                budget,
                workload_shape,
                execution.clone(),
            ),
        )
        .await
        {
            return response;
        }
    }
    // Do not wait beyond the deadline for a lock or perform fresh retrieval.
    // Source/selection changes or lock contention make the saved answer unusable.
    let local = execution.take_local().and_then(|(response, expected)| {
        let engine = state.engine.try_read().ok()?;
        (execution.authority_is_current(&engine) && local_checkpoint_is_current(&engine, &req, &expected)).then_some(response)
    });
    (
        StatusCode::OK,
        Json(execution.timeout_response(&req, local)),
    )
        .into_response()
}

pub(crate) fn local_checkpoint_is_current(
    engine: &MemoryEngine,
    req: &AnswerRequest,
    expected: &[RetrievedItem],
) -> bool {
    if validate_explicit_answer_selection(engine, req).is_err() {
        return false;
    }
    let ids = expected.iter().map(|item| item.id).collect::<Vec<_>>();
    if ids.iter().any(|id| {
        !engine.memories.iter().any(|memory| {
            memory.id == *id
                && crate::memory_engine_summary::memory_record_is_eligible_for_selected_answer(
                    memory,
                )
                && req
                    .conversation_id
                    .as_deref()
                    .is_none_or(|scope| memory.conversation_id == scope)
                && req
                    .selected_memory_ids
                    .as_ref()
                    .is_none_or(|selected| selected.contains(id))
        })
    }) {
        return false;
    }
    let current = engine.materialize_items(&ids, &req.query);
    reasoning_sources_unchanged(expected, &current)
}

pub(crate) async fn answer_memory_stream_http(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    headers: HeaderMap,
    Json(req): Json<AnswerRequest>,
) -> Response {
    let (budget, workload) = execution_budget_from_headers(
        &headers,
        crate::answer_execution::answer_deadline_ms(req.reasoning_mode.unwrap_or_default()),
    );
    let execution = Arc::new(crate::answer_execution::AnswerExecution::new(
        req.reasoning_mode.unwrap_or_default(),
        &budget,
    ));
    if let Err(response) = validate_answer_shape(&req) {
        return response;
    }
    // Reject invalid selections before committing SSE headers. The actual
    // shared service validates again before reading or using evidence.
    match tokio::time::timeout_at(execution.deadline, state.engine.read()).await {
        Ok(engine) => {
            if let Err(response) = validate_explicit_answer_selection(&engine, &req) {
                return response;
            }
        }
        Err(_) => {
            return (StatusCode::OK, Json(execution.timeout_response(&req, None))).into_response()
        }
    }
    let progress = execution.subscribe();
    let work = answer_memory_with_control(state, auth, req, budget, workload, execution.clone());
    axum::response::Sse::new(answer_event_stream(work, progress, execution))
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

fn answer_event_stream(
    work: impl std::future::Future<Output = Response> + Send + 'static,
    progress: tokio::sync::mpsc::Receiver<serde_json::Value>,
    execution: Arc<crate::answer_execution::AnswerExecution>,
) -> impl futures_util::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> + Send
{
    struct StreamState {
        work: std::pin::Pin<Box<dyn std::future::Future<Output = Response> + Send>>,
        progress: tokio::sync::mpsc::Receiver<serde_json::Value>,
        execution: Arc<crate::answer_execution::AnswerExecution>,
        result: Option<serde_json::Value>,
        finished: bool,
    }
    let state = StreamState {
        work: Box::pin(work),
        progress,
        execution,
        result: None,
        finished: false,
    };
    futures_util::stream::unfold(state, |mut state| async move {
        if state.finished {
            return None;
        }
        loop {
            // Stages produced in one poll are drained before the final result.
            if let Ok(progress) = state.progress.try_recv() {
                let event = axum::response::sse::Event::default()
                    .event("stage")
                    .data(progress.to_string());
                return Some((Ok(event), state));
            }
            if let Some(result) = state.result.take() {
                state.finished = true;
                let event = axum::response::sse::Event::default()
                    .event("result")
                    .data(result.to_string());
                return Some((Ok(event), state));
            }
            tokio::select! {
                Some(progress) = state.progress.recv() => {
                    let event = axum::response::sse::Event::default().event("stage").data(progress.to_string());
                    return Some((Ok(event), state));
                },
                response = &mut state.work => {
                    let status = response.status().as_u16();
                    let response = match axum::body::to_bytes(response.into_body(), usize::MAX).await {
                        Ok(bytes) => serde_json::from_slice::<serde_json::Value>(&bytes)
                            .unwrap_or_else(|_| serde_json::json!({"error_code":"answer_stream_invalid_json"})),
                        Err(_) => serde_json::json!({"error_code":"answer_stream_body_failed"}),
                    };
                    state.result = Some(serde_json::json!({
                        "http_status":status, "response":response,
                        "elapsed_ms":state.execution.started.elapsed().as_millis(),
                        "calls":state.execution.calls.load(std::sync::atomic::Ordering::Relaxed),
                        "mode":state.execution.mode
                    }));
                }
            }
        }
    })
}

async fn answer_memory_until(
    state: AppState,
    auth: Option<Extension<AuthContext>>,
    req: AnswerRequest,
    budget: ExecutionBudget,
    workload_shape: String,
    execution: Arc<crate::answer_execution::AnswerExecution>,
) -> Response {
    if let Err(response) = validate_answer_shape(&req) {
        return response;
    }
    let candidate_prefetch_timeout_ms = budget
        .timeout_ms
        .unwrap_or(5_000)
        .saturating_div(3)
        .clamp(250, 1_500);
    let mut performance = RequestPerformanceBuilder::new("memory_answer", &workload_shape, budget);
    let admission_started = performance.stage_start();
    let eligible = {
        let engine = state.engine.read().await;
        if let Err(response) = validate_explicit_answer_selection(&engine, &req) {
            return response;
        }
        execution.freeze_authority(&engine);
        req.selected_memory_ids
            .as_ref()
            .map(Vec::len)
            .unwrap_or_else(|| {
                engine
                    .memories
                    .iter()
                    .filter(|memory| {
                        req.conversation_id
                            .as_deref()
                            .map(|space_id| memory.conversation_id == space_id)
                            .unwrap_or(true)
                    })
                    .count()
            })
    };
    performance.record_stage(
        "request_admission",
        "core_interaction",
        "memory_service",
        admission_started,
        "completed",
        None,
    );

    let provider_requested = req.allow_external_provider == Some(true)
        && req.draft_answer.is_none()
        && !matches!(req.response_mode.as_deref(), Some("evidence" | "verbatim"));
    let mut reasoning_degradation = None;
    let mut attempted_reasoning_calls = 0;
    let mut attempted_reasoning_usage = (crate::llm_protocol::TokenUsage::default(), 0usize);
    let reasoning_started = std::time::Instant::now();
    let mut reasoning = None;
    // Under fill_gaps the model only answers when the local answer is
    // incomplete, so question planning waits until that is known: a complete
    // local answer then costs no provider call at all.
    let prefer_model = crate::answer_reasoning_provider::model_answer_policy_prefers_model();
    // The small-model tier never plans, so it always defers its one call.
    let small_tier = crate::answer_reasoning_provider::small_model_tier();
    let mut deferred_session = None;
    if provider_requested && (!prefer_model || small_tier) {
        match crate::answer_reasoning_provider::ReasoningSession::configured(
            req.reasoning_mode.unwrap_or_default(),
        ) {
            Ok(session) => deferred_session = Some(session.with_execution(execution.clone())),
            Err(error) => reasoning_degradation = Some(error),
        }
    } else if provider_requested {
        execution.stage("question_planning");
        let started = performance.stage_start();
        match crate::answer_reasoning_provider::ReasoningSession::configured(
            req.reasoning_mode.unwrap_or_default(),
        ) {
            Ok(session) => {
                let mut session = session.with_execution(execution.clone());
                match session.question(state.as_ref(), &req).await {
                    Ok(plan) => reasoning = Some((session, plan)),
                    Err(error) => {
                        attempted_reasoning_calls = session.calls;
                        attempted_reasoning_usage = (session.usage, session.usage_unreported);
                        reasoning_degradation = Some(error);
                    }
                }
            }
            Err(error) => reasoning_degradation = Some(error),
        }
        performance.record_stage(
            "question_planning",
            "core_interaction",
            "external_answer_provider",
            started,
            if reasoning.is_some() {
                "completed"
            } else {
                "degraded"
            },
            reasoning_degradation.clone(),
        );
    }
    execution.stage("continuity_refresh");
    let continuity_started = performance.stage_start();
    let auto_refresh_note = if req.selected_memory_ids.is_some() {
        None
    } else {
        maybe_auto_refresh_long_document_summary_for_answer(
            &state,
            auth.as_ref().map(|Extension(auth)| auth),
            &req,
        )
        .await
    };
    performance.record_stage(
        "continuity_refresh",
        "continuity",
        "long_document_summary",
        continuity_started,
        "completed",
        auto_refresh_note
            .as_ref()
            .map(|_| "source_summary_refreshed".to_string()),
    );
    let answer_started = performance.stage_start();
    execution.stage("candidate_prefetch");
    let prefetch_started = performance.stage_start();
    let planned_retrieval_query = reasoning.as_ref().map(|(_, plan)| plan.retrieval_query(&req.query));
    let candidate_cache_context=serde_json::json!({"mode":req.reasoning_mode.unwrap_or_default(),
        "external_provider_approved":req.allow_external_provider==Some(true),
        "question_plan":reasoning.as_ref().map(|(_,plan)|plan),
        "provider_identity":reasoning.as_ref().map(|(session,_)|session.cache_identity())});
    let candidate_limit = crate::memory_engine_summary::answer_candidate_limit(&req, eligible);
    let (prefetched_ids, prefetch_degraded_reason) = if req.selected_memory_ids.is_none() {
        let top_k = candidate_limit;
        let prefetch = tokio::time::timeout(
            std::time::Duration::from_millis(candidate_prefetch_timeout_ms),
            crate::support::retrieve_answer_candidates(
                state.clone(),
                QueryRequest {
                    conversation_id: req.conversation_id.clone(),
                    query: planned_retrieval_query.clone().unwrap_or_else(|| req.query.clone()),
                    top_k: Some(top_k),
                },
                &candidate_cache_context,
                auth.as_ref().map(|auth| &auth.0),
            ),
        )
        .await;
        match prefetch {
            Ok(Some(response)) => {
                performance.record_stage(
                    "candidate_prefetch",
                    "core_interaction",
                    "postgresql_retrieval",
                    prefetch_started,
                    "completed",
                    None,
                );
                (
                    Some(
                        response
                            .items
                            .into_iter()
                            .map(|item| item.id)
                            .collect::<Vec<_>>(),
                    ),
                    None,
                )
            }
            Ok(None) => {
                performance.record_stage(
                    "candidate_prefetch",
                    "core_interaction",
                    "postgresql_retrieval",
                    prefetch_started,
                    "fallback",
                    Some("in_memory_retrieval_required".to_string()),
                );
                (None, None)
            }
            Err(_) => {
                let reason = format!(
                    "PostgreSQL candidate prefetch exceeded {candidate_prefetch_timeout_ms} ms; the answer used bounded in-memory retrieval."
                );
                performance.record_stage(
                    "candidate_prefetch",
                    "core_interaction",
                    "postgresql_retrieval",
                    prefetch_started,
                    "degraded",
                    Some("prefetch_timeout_in_memory_fallback".to_string()),
                );
                performance.record_degradation(crate::api_types::ComponentDegradationReceipt {
                    component: "answer_candidate_prefetch".to_string(),
                    observed_status: "timeout".to_string(),
                    disposition: "fallback_completed".to_string(),
                    fallback_from: "postgresql_joint_retrieval".to_string(),
                    fallback_to: "bounded_in_memory_retrieval".to_string(),
                    reason_code: "prefetch_timeout_in_memory_fallback".to_string(),
                    retry_count: 0,
                    user_visible: true,
                });
                (None, Some(reason))
            }
        }
    } else {
        performance.record_stage(
            "candidate_prefetch",
            "core_interaction",
            "exact_memory_selection",
            prefetch_started,
            "skipped",
            Some("selected_memory_ids_supplied".to_string()),
        );
        (None, None)
    };
    let use_full_engine_snapshot =
        crate::answer_pipeline::requires_full_space_answer_records(&req.query);
    let candidate_freeze_started = performance.stage_start();
    execution.stage("candidate_freeze");
    let answer_engine = {
        let engine = state.engine.read().await;
        // A PG candidate prefetch is only a ranking hint. For a non-explicit
        // answer, widen the bounded worker snapshot with every pinned
        // activation that already passed conversation, visibility, liveness,
        // and ordinary answer-eligibility checks. Explicit selections remain
        // exact and are never expanded here.
        let forced_pinned_ids = if req.selected_memory_ids.is_none() && !use_full_engine_snapshot {
            engine.pinned_activation_ids_for_scope(req.conversation_id.as_deref(), None)
        } else {
            Vec::new()
        };
        let nominated_ids = if req.selected_memory_ids.is_none() && !use_full_engine_snapshot {
            prefetched_ids.as_ref().map(|ids| {
                let mut merged = ids.to_vec();
                let mut seen = merged.iter().copied().collect::<HashSet<_>>();
                for id in forced_pinned_ids {
                    if seen.insert(id) {
                        merged.push(id);
                    }
                }
                merged
            })
        } else {
            None
        };
        let nominated = req
            .selected_memory_ids
            .as_deref()
            .or_else(|| nominated_ids.as_deref());
        engine.answer_snapshot_for_scope(req.conversation_id.as_deref(), nominated)
    };
    performance.record_stage(
        "candidate_freeze",
        "core_interaction",
        "memory_engine_snapshot",
        candidate_freeze_started,
        "completed",
        None,
    );
    let answer_req = req.clone();
    // A remote semantic scorer sends evidence text through provider egress,
    // which needs this request's workspace scope and provider approval.
    let scorer_scope = if crate::semantic_scorer::scorer_requires_request_scope() {
        Some(
            crate::embedding_scope_for_conversation(
                &state,
                req.conversation_id.as_deref(),
                req.allow_external_provider == Some(true),
            )
            .await,
        )
    } else {
        None
    };
    let claim_planning_started = performance.stage_start();
    execution.stage("claim_planning");
    execution.stage("local_answer_worker");
    let mut worker =
        crate::answer_execution::BlockingAnswerWorker::spawn(execution.deadline, move || {
          crate::semantic_scorer::with_request_scope(scorer_scope, move || {
            let mut engine = answer_engine;
            let prefetched_ids = if prefetched_ids.is_none() && answer_req.selected_memory_ids.is_none() {
                planned_retrieval_query.as_ref().map(|query|
                    engine.retrieve_ids(answer_req.conversation_id.as_deref(), query, candidate_limit))
            } else { prefetched_ids };
            let (response, ids) =
                engine.answer_with_self_check_prefetched(&answer_req, prefetched_ids.as_deref());
            let sources = engine.materialize_items(&ids, &answer_req.query);
            let checkpoint_ids = response
                .evidence
                .iter()
                .map(|item| item.id)
                .collect::<Vec<_>>();
            let checkpoint_sources = engine.materialize_items(&checkpoint_ids, &answer_req.query);
            (response, ids, sources, checkpoint_sources)
          })
        });
    let answer_execution = (&mut worker.0).await;
    let (mut resp, mut retrieved_ids, reasoning_sources, checkpoint_sources) =
        match answer_execution {
            Ok(Ok(result)) => result,
            Ok(Err(_)) => return (StatusCode::OK, Json(execution.timeout_response(&req, None))).into_response(),
            Err(error) => {
                performance.record_stage(
                    "retrieval_rerank_context_assembly",
                    "core_interaction",
                    "memory_engine",
                    answer_started,
                    "failed",
                    Some("answer_worker_failed".to_string()),
                );
                return api_error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "answer_execution_failed",
                    "answer_worker_failed",
                    "the answer worker could not complete this request",
                    serde_json::json!({"reason": error.to_string()}),
                )
                .into_response();
            }
        };
    execution.save_local(&resp, checkpoint_sources);
    performance.record_stage(
        "claim_planning",
        "core_interaction",
        "answer_pipeline",
        claim_planning_started,
        "completed",
        None,
    );
    let independent_verification_started = performance.stage_start();
    execution.stage("independent_verification");
    if let Some((_, plan)) = reasoning.as_ref() {
        resp.semantic_notes.push(SemanticAnswerNote {
            note_type:"question_plan_retrieval".into(),
            message:serde_json::json!({"query":plan.retrieval_query(&req.query),
                "original_question_unchanged":true,
                "selection":if req.selected_memory_ids.is_some() {"explicit_ids_no_expansion"} else {"typed_plan_candidate_search"},
                "authority":"retrieval_hint_not_verified_fact"}).to_string(),
            entity_id:None,attribute_key:None,scope:req.conversation_id.clone(),candidate_values:vec![],
        });
    }
    if let (Some(pg), Some(scope)) = (state.pg.as_ref(), req.conversation_id.as_deref()) {
        let engine = state.engine.read().await;
        if local_checkpoint_is_current(&engine, &req, &reasoning_sources) {
            let ids = reasoning_sources.iter().map(|item| item.id).collect::<HashSet<_>>();
            let records = engine.memories.iter().filter(|record|
                record.conversation_id == scope && ids.contains(&record.id))
                .cloned().collect::<Vec<_>>();
            execution.stage("evidence_projection_refresh");
            let (status, atoms, deferred) = match tokio::time::timeout_at(
                execution.deadline.min(tokio::time::Instant::now() + std::time::Duration::from_millis(500)),
                pg.ensure_answer_evidence_projections_bounded(&records, 64),
            ).await {
                Ok(Ok((count, deferred))) => ("completed", count, Some(deferred)),
                Ok(Err(_)) => ("failed", 0, None),
                Err(_) => ("timeout", 0, None),
            };
            drop(engine);
            // Timeout/error may follow a committed database write. Invalidate
            // conservatively in those cases too; no-op success needs no epoch.
            if atoms > 0 || status != "completed" {
                state.engine.write().await.invalidate_cache_for_conversation(Some(scope));
            }
            resp.semantic_notes.push(SemanticAnswerNote {
                note_type:"evidence_projection_refresh".into(),
                message:serde_json::json!({"status":status,"atoms_written":atoms,
                    "candidate_records":records.len(),"deferred_records":deferred,
                    "rebuild_limit":64,"budget_applied_after_staleness_check":true,
                    "policy":"lazy_selected_sources_only_no_backfill",
                    "proof_pool_truncated":false}).to_string(),
                entity_id:None, attribute_key:None, scope:Some(scope.to_string()), candidate_values:vec![],
            });
        }
    }
    {
        let mut engine = state.engine.write().await;
        engine.mark_retrieved(&retrieved_ids);
    }
    if resp.selection_receipt.is_some() {
        resp.runtime_enforcement = serde_json::json!({
            "runtime_format": "gateway_runtime_enforcement_v1",
            "enforcement": "selection_scope_enforced",
            "entrypoint": req
                .selection_binding
                .as_ref()
                .map(|binding| binding.source.as_str())
                .unwrap_or("direct_memory_selection"),
            "conformance_status": "exact_selected_memory_scope_only",
            "bypass_visible": true,
            "reason": "the selected evidence scope is fail-closed; direct answer still does not grant write, external-action, or full managed-turn authority"
        });
    }
    let mut verified_delivery_sources = None;
    // A complete local answer is kept regardless of the model plan, so the
    // generation and verification calls would be wasted latency and cost.
    // A complete local answer that is a quoted sentence (not a typed value or
    // date arithmetic) is still checked by the selection tier for trick
    // premises; select_evidence skips questions not about one person.
    // A local date answer is checked too: it is kept when the model picks
    // the same turn (or nothing dated), and replaced when the model's dated
    // sentence is a different one ("career-high in assists" for "points").
    let local_date_answer = resp.answer_pipeline.as_ref().is_some_and(|receipt|
        receipt.operator_plan.selected_operator.as_deref() == Some("event_date_v1"));
    let verify_local = small_tier
        && resp.answer_status == "answered"
        && !prefer_model
        && (local_date_answer || resp.answer_pipeline.as_ref().is_some_and(|receipt| matches!(
            receipt.operator_plan.selected_operator.as_deref(),
            Some("unified_answer_generator_v1" | "focused_scalar_v1" | "grounded_list_v1"))));
    let local_turns = resp.claims.iter()
        .flat_map(|claim| claim.evidence_ids.iter().flat_map(move |id| claim.evidence_source_anchors.iter()
            .map(move |anchor| (*id, anchor.value.clone()))))
        .collect::<std::collections::BTreeSet<_>>();
    let reasoning = if resp.answer_status == "answered" && !prefer_model && !verify_local {
        if reasoning.is_some() || deferred_session.is_some() {
            resp.semantic_notes.push(SemanticAnswerNote {
                note_type: "reasoning_local_answer_kept".into(),
                message: "The locally verified answer is complete; model generation was skipped.".into(),
                entity_id: None,
                attribute_key: None,
                scope: req.conversation_id.clone(),
                candidate_values: Vec::new(),
            });
        }
        None
    } else if small_tier && deferred_session.is_some() {
        deferred_session.take().map(|session| (session, None))
    } else if let Some(mut session) = deferred_session {
        execution.stage("question_planning");
        let started = performance.stage_start();
        let planned = match session.question(state.as_ref(), &req).await {
            Ok(plan) => Some((session, Some(plan))),
            Err(error) => {
                attempted_reasoning_calls = session.calls;
                attempted_reasoning_usage = (session.usage, session.usage_unreported);
                reasoning_degradation = Some(error);
                None
            }
        };
        performance.record_stage(
            "question_planning",
            "core_interaction",
            "external_answer_provider",
            started,
            if planned.is_some() { "completed" } else { "degraded" },
            reasoning_degradation.clone(),
        );
        planned
    } else {
        reasoning.map(|(session, plan)| (session, Some(plan)))
    };
    if let Some((mut session, plan)) = reasoning {
        let provider_started = performance.stage_start();
        let temporal_question = resp.answer_pipeline.as_ref()
            .is_some_and(|receipt| receipt.query_plan.answer_target.eq_ignore_ascii_case("temporal"));
        let outcome = match &plan {
            Some(plan) => session.answer(state.as_ref(), &req, plan, &reasoning_sources).await,
            None => {
                let judgment = resp.answer_pipeline.as_ref()
                    .is_some_and(|receipt| receipt.query_plan.inference_type != "direct");
                let list_question = resp.answer_pipeline.as_ref()
                    .is_some_and(|receipt| receipt.query_plan.cardinality == "list");
                session.select_evidence(state.as_ref(), &req, &reasoning_sources, temporal_question, judgment, verify_local, list_question).await
            }
        };
        match outcome
        {
            Ok(mut answer) => {
                let snapshot = std::mem::take(&mut answer.source_snapshot);
                let engine = state.engine.read().await;
                if execution.authority_is_current(&engine) && session.authority_is_current(&engine, &req)
                    && local_checkpoint_is_current(&engine, &req, &snapshot)
                {
                    if let (Some(pg), Some(scope), Some(proof)) = (
                        state.pg.as_ref(), req.conversation_id.as_deref(), answer.proof_projection.as_ref(),
                    ) {
                        let workspace = engine.workspace_for_conversation(scope);
                        let tenant = engine.workspace_tenant_id(&workspace);
                        let ids = snapshot.iter().map(|item| item.id).collect::<HashSet<_>>();
                        let records = engine.memories.iter().filter(|record| ids.contains(&record.id))
                            .cloned().collect::<Vec<_>>();
                        execution.stage("proof_projection_persist");
                        // Keep the authority/source read guard while saving, and
                        // remain inside the original request's deadline.
                        let status = match tokio::time::timeout_at(
                            execution.deadline.min(tokio::time::Instant::now() + std::time::Duration::from_millis(500)),
                            pg.persist_answer_reasoning_proof(scope, &tenant, &workspace, proof, &records),
                        ).await {
                            Ok(Ok(true)) => "persisted",
                            Ok(Ok(false)) => "not_written_conflict_or_changed_source",
                            Ok(Err(_)) => "persistence_failed",
                            Err(_) => "persistence_timeout",
                        };
                        resp.semantic_notes.push(SemanticAnswerNote {
                            note_type: "reasoning_proof_projection".into(),
                            message: serde_json::json!({"status":status,"proof_id":proof.proof_id,
                                "storage_id":proof.scoped_storage_id(scope, &tenant, &workspace).ok(),
                                "reuse":"audit_only_not_answer_cache"}).to_string(),
                            entity_id:None, attribute_key:None, scope:Some(scope.to_string()), candidate_values:vec![],
                        });
                    }
                    // A model plan with no independently verified claim does
                    // not overrule a locally verified answer; it only adds
                    // answers the local extractive path could not produce.
                    // A complete local answer (verified extraction, date
                    // arithmetic) is kept; the model fills questions the local
                    // path left partial or open. Measured on the LoCoMo dev
                    // sample, letting model plans replace local answers added
                    // more wrong answers than right ones.
                    // The model judged the premise false: withhold any local
                    // answer too (withholding never adds an unverified claim).
                    // For a local date answer the call only decides whether a
                    // different dated sentence replaces it: measured on agnes, a
                    // premise rejection there withheld correct dates.
                    if answer.premise_rejected && !local_date_answer {
                        resp.semantic_notes.push(SemanticAnswerNote {
                            note_type: "premise_not_supported".into(),
                            message: "The conversation does not state this about the person asked; no answer is given.".into(),
                            entity_id: None,
                            attribute_key: None,
                            scope: req.conversation_id.clone(),
                            candidate_values: answer.gaps.clone(),
                        });
                        apply_verified_reasoning_answer(&mut resp, answer, &snapshot);
                    } else if (resp.answer_status == "answered" && !prefer_model && !(verify_local && !answer.claims.is_empty()
                            && !(local_date_answer && answer.claims.iter().flat_map(|claim| claim.atoms.iter())
                                .any(|atom| atom.source_anchor.as_ref().is_some_and(|anchor|
                                    local_turns.contains(&(atom.memory_id, anchor.value.clone())))))))
                        || (answer.claims.is_empty()
                            && matches!(resp.answer_status.as_str(), "answered" | "partial"))
                    {
                        resp.semantic_notes.push(SemanticAnswerNote {
                            note_type: "reasoning_local_answer_kept".into(),
                            message: "The locally verified answer is kept; the model plan is not used.".into(),
                            entity_id: None,
                            attribute_key: None,
                            scope: req.conversation_id.clone(),
                            candidate_values: answer.gaps.clone(),
                        });
                    } else {
                        retrieved_ids.extend(snapshot.iter().map(|item| item.id));
                        retrieved_ids.sort_unstable();
                        retrieved_ids.dedup();
                        apply_verified_reasoning_answer(&mut resp, answer, &snapshot);
                        verified_delivery_sources = Some(snapshot);
                    }
                } else {
                    reasoning_degradation = Some("source_snapshot_changed_during_reasoning".into());
                    apply_verified_reasoning_answer(
                        &mut resp,
                        crate::answer_reasoning_provider::VerifiedAnswer {
                            claims: Vec::new(),
                            gaps: vec!["source_snapshot_changed_during_reasoning".into()],
                            source_snapshot: Vec::new(),
                            admission: None,
                            proof_projection: None,
                            verification: crate::answer_reasoning_provider::VerificationMethod::SeparateModelCall,
                            premise_rejected: false,
                        },
                        &[],
                    );
                }
            }
            Err(error) => {
                reasoning_degradation = Some(error);
                let engine = state.engine.read().await;
                if !execution.authority_is_current(&engine) || !session.authority_is_current(&engine, &req)
                    || !local_checkpoint_is_current(&engine, &req, &reasoning_sources)
                {
                    apply_verified_reasoning_answer(
                        &mut resp,
                        crate::answer_reasoning_provider::VerifiedAnswer {
                            claims: vec![],
                            gaps: vec!["local_source_snapshot_changed".into()],
                            source_snapshot: vec![],
                            admission: None,
                            proof_projection: None,
                            verification: crate::answer_reasoning_provider::VerificationMethod::SeparateModelCall,
                            premise_rejected: false,
                        },
                        &[],
                    );
                }
            }
        }
        performance.record_stage(
            "semantic_answer_verification",
            "core_interaction",
            "external_answer_provider",
            provider_started,
            if reasoning_degradation.is_some() {
                "degraded"
            } else {
                "completed"
            },
            reasoning_degradation.clone(),
        );
        resp.semantic_notes.push(SemanticAnswerNote {
            note_type: "reasoning_execution".into(),
            message: serde_json::json!({"contract":crate::answer_reasoning_contract::CONTRACT_VERSION,
                "mode":session.mode,"model":session.model,"calls":session.calls,"elapsed_ms":session.elapsed_ms(),
                "usage":session.usage,"usage_unreported_calls":session.usage_unreported,
                "verification_method":if plan.is_some() { "separate_call_same_configured_model" } else { "verbatim_quotation_deterministic_checks" },
                "tier":if plan.is_some() { "full" } else { "small" }}).to_string(),
            entity_id: None, attribute_key: None, scope: req.conversation_id.clone(), candidate_values: Vec::new(),
        });
    }
    performance.record_stage(
        "independent_verification",
        "core_interaction",
        if provider_requested { "answer_verifier" } else { "local_claim_verifier" },
        independent_verification_started,
        if reasoning_degradation.is_some() { "degraded" } else { "completed" },
        reasoning_degradation.clone(),
    );
    if let Some(reason) = reasoning_degradation {
        resp.semantic_notes.push(SemanticAnswerNote {
            note_type: "semantic_answer_provider_degraded".into(),
            message: reason,
            entity_id: None,
            attribute_key: None,
            scope: req.conversation_id.clone(),
            candidate_values: Vec::new(),
        });
    }
    if !resp
        .semantic_notes
        .iter()
        .any(|note| note.note_type == "reasoning_execution")
    {
        resp.semantic_notes.push(SemanticAnswerNote {
            note_type: "reasoning_execution".into(),
            message: serde_json::json!({"requested_mode":req.reasoning_mode.unwrap_or_default(),
                "mode":req.reasoning_mode.unwrap_or_default(),
                "verification_method":"local_pipeline_only",
                "calls":attempted_reasoning_calls,
                "usage":attempted_reasoning_usage.0,"usage_unreported_calls":attempted_reasoning_usage.1,
                "elapsed_ms":reasoning_started.elapsed().as_millis(),
                "provider_requested":provider_requested,
                "note":"No verified model execution receipt available; local content is not model verified."
            }).to_string(),
            entity_id:None, attribute_key:None, scope:req.conversation_id.clone(), candidate_values:Vec::new(),
        });
    }
    performance.record_stage(
        "retrieval_rerank_context_assembly",
        "core_interaction",
        "memory_engine",
        answer_started,
        "completed",
        None,
    );
    if let Some(note) = auto_refresh_note {
        resp.semantic_notes.push(note);
    }
    if let Some(message) = prefetch_degraded_reason {
        resp.semantic_notes.push(SemanticAnswerNote {
            note_type: "retrieval_prefetch_degraded".to_string(),
            message,
            entity_id: None,
            attribute_key: None,
            scope: req.conversation_id.clone(),
            candidate_values: Vec::new(),
        });
    }
    performance.mark_first_useful();

    execution.stage("verified_delivery");
    let request_id = next_memory_trace_id();

    // Build the trace from answer evidence
    let trace_assembly_started = performance.stage_start();
    let mut used_ids = std::collections::HashSet::new();
    for cov in &resp.coverage {
        for id in &cov.evidence_ids {
            used_ids.insert(*id);
        }
    }
    // An answer with no verified claim evidence remains ungrounded. Do not
    // promote nearby memories merely to make the trace look populated.

    execution.stage("source_citation_resolution");
    let citation_started = performance.stage_start();
    if let Some(pg) = &state.pg {
        let evidence_ids = resp
            .evidence
            .iter()
            .filter(|item| used_ids.contains(&item.id))
            .map(|item| item.id)
            .collect::<Vec<_>>();
        match pg.universal_citations_for_memory_ids(&evidence_ids).await {
            Ok(citations) => {
                let mut by_memory_id = citations
                    .into_iter()
                    .map(|citation| (citation.memory_id, citation))
                    .collect::<HashMap<_, _>>();
                resp.citations = evidence_ids
                    .iter()
                    .filter_map(|memory_id| by_memory_id.remove(memory_id))
                    .collect();
                performance.record_stage(
                    "source_citation_resolution",
                    "source_provenance",
                    "postgresql_universal_source_links",
                    citation_started,
                    "completed",
                    None,
                );
            }
            Err(error) => {
                performance.record_stage(
                    "source_citation_resolution",
                    "source_provenance",
                    "postgresql_universal_source_links",
                    citation_started,
                    "degraded",
                    Some(error.clone()),
                );
                resp.semantic_notes.push(SemanticAnswerNote {
                    note_type: "source_citation_degraded".to_string(),
                    message: "Structured source citations could not be resolved for this answer."
                        .to_string(),
                    entity_id: None,
                    attribute_key: None,
                    scope: req.conversation_id.clone(),
                    candidate_values: Vec::new(),
                });
            }
        }
    } else {
        performance.record_stage(
            "source_citation_resolution",
            "source_provenance",
            "memory_only_backend",
            citation_started,
            "degraded",
            Some("PostgreSQL source links are unavailable".to_string()),
        );
    }

    let claim_evidence_ids = used_ids.clone();
    let response_evidence_ids = resp
        .evidence
        .iter()
        .map(|item| item.id)
        .collect::<HashSet<_>>();
    let mut citation_memory_ids = resp
        .citations
        .iter()
        .map(|citation| citation.memory_id)
        .collect::<HashSet<_>>();
    citation_memory_ids.extend(crate::memory_engine_summary::cited_memory_ids(&resp.answer));
    let explicit_evidence_mode = matches!(
        req.response_mode.as_deref(),
        Some("evidence") | Some("verbatim")
    );
    let citation_binding_applicable = !claim_evidence_ids.is_empty() && !explicit_evidence_mode;
    let strict_citation_bound = citation_binding_applicable
        && claim_evidence_ids == response_evidence_ids
        && claim_evidence_ids.is_subset(&citation_memory_ids);
    let mut final_evidence_ids = claim_evidence_ids.iter().copied().collect::<Vec<_>>();
    final_evidence_ids.sort_unstable();

    if let Some(pipeline) = resp.answer_pipeline.as_mut() {
        pipeline.final_evidence_ids = final_evidence_ids.clone();
        if let Some(stage) = pipeline
            .stage_trace
            .iter_mut()
            .find(|stage| stage.stage == "citation_binding")
        {
            stage.status = if !citation_binding_applicable {
                "not_applicable"
            } else if strict_citation_bound {
                "completed"
            } else {
                "failed"
            }
            .to_string();
            stage.input_count = claim_evidence_ids.len();
            stage.output_count = citation_memory_ids
                .intersection(&claim_evidence_ids)
                .count();
            stage.evidence_ids = final_evidence_ids.clone();
            stage.reason = if !citation_binding_applicable {
                Some("no generated claim requires citation binding".to_string())
            } else if strict_citation_bound {
                None
            } else {
                Some(format!(
                    "claim_ids={}, response_evidence_ids={}, cited_claim_ids={}",
                    claim_evidence_ids.len(),
                    response_evidence_ids.len(),
                    citation_memory_ids
                        .intersection(&claim_evidence_ids)
                        .count()
                ))
            };
        }
        if citation_binding_applicable && !strict_citation_bound {
            pipeline.failure_class = "citation_filter_mismatch".to_string();
        }
    }
    if citation_binding_applicable && !strict_citation_bound {
        if resp.answer_status == "answered" {
            resp.answer_status = "uncertain".to_string();
        }
        resp.semantic_notes.push(SemanticAnswerNote {
            note_type: "citation_binding".to_string(),
            message: "The answer was downgraded because every verified claim could not be bound to the same response evidence and source citations.".to_string(),
            entity_id: None,
            attribute_key: None,
            scope: req.conversation_id.clone(),
            candidate_values: final_evidence_ids.iter().map(u64::to_string).collect(),
        });
    }

    let injected_evidence = resp
        .evidence
        .iter()
        .filter(|item| used_ids.contains(&item.id))
        .map(|item| MemoryContextEvidence {
            memory_id: item.id,
            layer: item.layer.clone(),
            memory_type: item.memory_type.clone(),
            score: item.score,
            source: item.source.clone(),
            created_at: item.created_at,
            reason: working_context_reason(item),
            content: item.content.clone(),
        })
        .collect::<Vec<_>>();

    let memories_searched = resp
        .evidence
        .iter()
        .map(|item| trace_item_from_retrieved(item, used_ids.contains(&item.id)))
        .collect::<Vec<_>>();
    let memories_injected = injected_evidence
        .iter()
        .map(trace_item_from_evidence)
        .collect::<Vec<_>>();
    let memories_selected = memories_injected.clone();
    let memories_ignored = resp
        .evidence
        .iter()
        .filter(|item| !used_ids.contains(&item.id))
        .map(|item| {
            let mut trace = trace_item_from_retrieved(item, false);
            trace.ignored_reason = Some(working_context_ignore_reason(item));
            trace
        })
        .collect::<Vec<_>>();

    let conflicts_or_stale_records = memories_searched
        .iter()
        .zip(resp.evidence.iter())
        .filter(|(_, source)| working_context_item_is_conflict_or_stale(source))
        .map(|(trace, _)| trace.clone())
        .collect::<Vec<_>>();

    let final_context = resp
        .evidence
        .iter()
        .filter(|item| used_ids.contains(&item.id))
        .map(|item| format!("- [{}] {}", item.memory_type, item.content))
        .collect::<Vec<_>>()
        .join("\n");

    let explanation = generate_trace_explanation(
        &memories_selected,
        &memories_ignored,
        &conflicts_or_stale_records,
        &final_context,
    );

    let memory_used_val = memories_selected.len();
    let memory_ignored_val = memories_ignored.len();
    let conflict_count_val = conflicts_or_stale_records.len();
    performance.record_stage(
        "trace_context_assembly",
        "current_understanding",
        "memory_trace",
        trace_assembly_started,
        "completed",
        None,
    );

    let answer_pipeline_trace = resp
        .answer_pipeline
        .as_ref()
        .and_then(|pipeline| serde_json::to_value(pipeline).ok());
    let trace = MemoryTraceResponse {
        request_id: request_id.clone(),
        trace_format: "memory_trace_v1".to_string(),
        generated_at: Utc::now(),
        memory_mode: "auto".to_string(),
        memory_status: if resp.answer_status == "answered"
            && !resp.claims.is_empty() && !resp.coverage.is_empty()
            && resp.coverage.iter().all(|c| c.covered) && strict_citation_bound {
            "fresh".to_string()
        } else {
            "stale".to_string()
        },
        decision_reason: format!("answer_status={}; answer_mode={}; {}; not an independent semantic quality judgment",
            resp.answer_status,resp.answer_mode,
            if resp.answer_mode=="evidence_only" {"source_browsing_not_claim_verification"}
            else {"nonempty_claim_coverage_and_citation_binding_required"}),
        recall_policy: None,
        policy_source: None,
        recall_outcome: None,
        reasoning_policy: None,
        vector_retrieval: None,
        memories_searched,
        memories_selected,
        memories_injected,
        memories_ignored,
        conflicts_or_stale_records,
        file_evidence: Vec::new(),
        final_context,
        explanation: Some(explanation),
        answer_pipeline: answer_pipeline_trace,
    };

    execution.stage("trace_store");
    let trace_store_started = performance.stage_start();
    let final_sources=verified_delivery_sources.as_deref().unwrap_or(&reasoning_sources);
    let trace_warning=match crate::support::store_current_answer_trace(
        &state,trace,&req,final_sources,&execution).await {
        Ok(warning)=>warning,
        Err(code)=> {
            let status=match code {"answer_authority_changed"=>StatusCode::FORBIDDEN,
                "answer_source_snapshot_changed"=>StatusCode::CONFLICT,_=>StatusCode::SERVICE_UNAVAILABLE};
            return api_error(status,code,code,"Answer state changed or is unavailable; retry against the current snapshot.",serde_json::json!({})).into_response();
        }
    };
    if let Some(reason)=trace_warning {
        resp.semantic_notes.push(SemanticAnswerNote{note_type:"trace_persistence_degraded".into(),
            message:reason.into(),entity_id:None,attribute_key:None,scope:req.conversation_id.clone(),candidate_values:vec![]});
    }
    performance.record_stage(
        "complete_result",
        "core_interaction",
        if state.pg.is_some() {
            "postgresql_trace_store"
        } else {
            "in_memory_trace_store"
        },
        trace_store_started,
        if trace_warning.is_some() {"degraded"} else {"completed"},
        trace_warning.map(str::to_string),
    );

    // Populate counts and trace_id in resp
    resp.memory_used = memory_used_val;
    resp.memory_ignored = memory_ignored_val;
    resp.conflict_count = conflict_count_val;
    resp.trace_id = request_id;
    performance.set_counts(WorkDispositionCounts {
        eligible,
        selected: memory_used_val,
        omitted: eligible.saturating_sub(memory_used_val),
        failed: conflict_count_val,
        ..WorkDispositionCounts::default()
    });
    if state.pg.is_none() {
        if let Ok(receipt) = component_degradation_receipt("database", "not_configured", false, 0) {
            performance.record_degradation(receipt);
        }
    }
    let render_started = performance.stage_start();
    performance.record_stage(
        "render_result",
        "core_interaction",
        "http_json",
        render_started,
        "completed",
        None,
    );
    let authority_guard = state.engine.read().await;
    if !execution.authority_is_current(&authority_guard) {
        return api_error(StatusCode::FORBIDDEN,"answer_authority_changed","answer_authority_changed",
            "Answer permissions changed during execution; retry under the current authority.",serde_json::json!({})).into_response();
    }
    let final_sources=verified_delivery_sources.as_deref().unwrap_or(&reasoning_sources);
    if !local_checkpoint_is_current(&authority_guard, &req, final_sources) {
        return api_error(StatusCode::CONFLICT,"answer_source_snapshot_changed","answer_source_snapshot_changed",
            "Source evidence changed before delivery; retry against the current source snapshot.",serde_json::json!({})).into_response();
    }

    let verified_delivery_started = performance.stage_start();
    performance.record_stage(
        "verified_delivery",
        "core_interaction",
        "answer_response",
        verified_delivery_started,
        "completed",
        None,
    );
    resp.performance = Some(performance.finish("not_applicable"));
    state.stats.record_answer(resp.evidence.len());
    (StatusCode::OK, Json(resp)).into_response()
}

pub(crate) async fn reindex_memory(
    State(state): State<AppState>,
    auth: Option<Extension<AuthContext>>,
    Json(req): Json<ReindexRequest>,
) -> Response {
    let Some(pg) = &state.pg else {
        return api_error(
            StatusCode::BAD_REQUEST,
            "postgres_backend_required",
            "postgres_backend_required",
            "this endpoint requires postgres backend",
            serde_json::json!({}),
        )
        .into_response();
    };

    let target_provider = req
        .target_provider
        .clone()
        .unwrap_or_else(|| state.embedder.provider_label().to_string());
    let target_model = req
        .target_model
        .clone()
        .unwrap_or_else(|| state.embedder.model_label().to_string());
    let dry_run = req.dry_run.unwrap_or(false);
    let only_outdated = req.only_outdated_embeddings.unwrap_or(false);
    let limit = req.limit.unwrap_or(10_000).max(1);
    let reindex_conversation_id = req.conversation_id.clone();
    {
        let engine = state.engine.read().await;
        if admin_scope_policy(state.as_ref(), &engine, auth.as_ref().map(|Extension(a)| a))
            .is_some()
            && req.conversation_id.is_none()
        {
            drop(engine);
            return admin_scope_required_with_audit(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "reindex",
                "scope",
                "scope",
                None,
                None,
                None,
                "scoped admin key must specify conversation_id for reindex",
            )
            .await;
        }
        if let Some(response) = enforce_admin_runtime_write_for_conversation(
            state.as_ref(),
            &engine,
            auth.as_ref().map(|Extension(a)| a),
            req.conversation_id.as_deref(),
            "reindex",
        ) {
            let (tenant_id, workspace_id, conversation_id) =
                if let Some(conversation_id) = req.conversation_id.as_deref() {
                    let workspace_id = engine.workspace_for_conversation(conversation_id);
                    let tenant_id = engine.workspace_tenant_id(&workspace_id);
                    (
                        Some(tenant_id),
                        Some(workspace_id),
                        Some(conversation_id.to_string()),
                    )
                } else {
                    (None, None, None)
                };
            drop(engine);
            audit_runtime_write_denial_for_conversation(
                &state,
                auth.as_ref().map(|Extension(a)| a),
                "reindex",
                "conversation",
                req.conversation_id.as_deref().unwrap_or("scope"),
                tenant_id,
                workspace_id,
                conversation_id,
                "admin scope denied reindex".to_string(),
                "admin_scope_forbidden",
            )
            .await;
            return response;
        }
    }
    let selected = {
        let engine = state.engine.read().await;
        engine
            .memories
            .iter()
            .filter(|m| {
                req.conversation_id
                    .as_deref()
                    .map(|cid| m.conversation_id == cid)
                    .unwrap_or(true)
            })
            .filter(|m| {
                if !only_outdated {
                    return true;
                }
                m.embedding_status.as_deref() != Some("ready")
                    || m.embedding_provider.as_deref() != Some(target_provider.as_str())
                    || m.embedding_model.as_deref() != Some(target_model.as_str())
            })
            .take(limit)
            .cloned()
            .collect::<Vec<_>>()
    };

    if selected.is_empty() {
        let resp = ReindexResponse {
            dry_run,
            selected: 0,
            reindexed: 0,
            skipped: 0,
            backend: state.backend.clone(),
            embedding: state.embedding_mode.clone(),
            target_provider,
            target_model,
        };
        audit_runtime_write_applied_for_conversation(
            &state,
            auth.as_ref().map(|Extension(a)| a),
            "reindex",
            "conversation",
            reindex_conversation_id.as_deref().unwrap_or("global"),
            reindex_conversation_id.as_deref(),
            serde_json::json!({
                "dry_run": dry_run,
                "selected": 0,
                "reindexed": 0,
                "skipped": 0,
                "only_outdated_embeddings": only_outdated,
            }),
        )
        .await;
        return (
            StatusCode::OK,
            Json(serde_json::to_value(resp).unwrap_or_default()),
        )
            .into_response();
    }

    if dry_run {
        let resp = ReindexResponse {
            dry_run: true,
            selected: selected.len(),
            reindexed: 0,
            skipped: 0,
            backend: state.backend.clone(),
            embedding: state.embedding_mode.clone(),
            target_provider,
            target_model,
        };
        audit_runtime_write_applied_for_conversation(
            &state,
            auth.as_ref().map(|Extension(a)| a),
            "reindex",
            "conversation",
            reindex_conversation_id.as_deref().unwrap_or("global"),
            reindex_conversation_id.as_deref(),
            serde_json::json!({
                "dry_run": true,
                "selected": selected.len(),
                "reindexed": 0,
                "skipped": 0,
                "only_outdated_embeddings": only_outdated,
            }),
        )
        .await;
        return (
            StatusCode::OK,
            Json(serde_json::to_value(resp).unwrap_or_default()),
        )
            .into_response();
    }

    let mut reindexed = 0usize;
    let mut skipped = 0usize;
    let scoped_selected = {
        let engine = state.engine.read().await;
        let mut grouped = HashMap::<(String, String), Vec<MemoryRecord>>::new();
        for memory in &selected {
            let workspace_id = engine.workspace_for_conversation(&memory.conversation_id);
            let tenant_id = engine.workspace_tenant_id(&workspace_id);
            grouped
                .entry((tenant_id, workspace_id))
                .or_default()
                .push(memory.clone());
        }
        grouped
    };
    for ((tenant_id, workspace_id), workspace_memories) in scoped_selected {
        let embedding_scope = crate::embedding::EmbeddingScope::new(tenant_id, workspace_id, false);
        for chunk in workspace_memories.chunks(64) {
            let texts = chunk.iter().map(|m| m.content.clone()).collect::<Vec<_>>();
            let embeddings = match state
                .embedder
                .embed_many_scoped(&texts, &embedding_scope)
                .await
            {
                Ok(embeddings) => embeddings,
                Err(e) => {
                    eprintln!("reindex embedding failed, skip chunk: {e}");
                    skipped += chunk.len();
                    continue;
                }
            };
            for (idx, memory) in chunk.iter().enumerate() {
                let Some(emb) = embeddings.get(idx) else {
                    skipped += 1;
                    continue;
                };
                match pg
                    .upsert_memory(memory, emb, &target_provider, &target_model)
                    .await
                {
                    Ok(_) => {
                        reindexed += 1;
                        let mut engine = state.engine.write().await;
                        if let Some(existing) =
                            engine.memories.iter_mut().find(|m| m.id == memory.id)
                        {
                            existing.embedding_provider = Some(target_provider.clone());
                            existing.embedding_model = Some(target_model.clone());
                            existing.embedding_status = Some("ready".to_string());
                            existing.embedded_at = Some(Utc::now());
                        }
                    }
                    Err(_) => skipped += 1,
                }
            }
        }
    }

    let resp = ReindexResponse {
        dry_run: false,
        selected: selected.len(),
        reindexed,
        skipped,
        backend: state.backend.clone(),
        embedding: state.embedding_mode.clone(),
        target_provider,
        target_model,
    };
    {
        let mut engine = state.engine.write().await;
        if let Some(conversation_id) = req.conversation_id.as_deref() {
            engine.invalidate_cache_for_conversation(Some(conversation_id));
        } else {
            engine.invalidate_all_cache();
        }
    }
    audit_runtime_write_applied_for_conversation(
        &state,
        auth.as_ref().map(|Extension(a)| a),
        "reindex",
        "conversation",
        reindex_conversation_id.as_deref().unwrap_or("global"),
        reindex_conversation_id.as_deref(),
        serde_json::json!({
            "dry_run": false,
            "selected": resp.selected,
            "reindexed": resp.reindexed,
            "skipped": resp.skipped,
            "only_outdated_embeddings": only_outdated,
            "target_provider": resp.target_provider,
            "target_model": resp.target_model,
        }),
    )
    .await;
    (
        StatusCode::OK,
        Json(serde_json::to_value(resp).unwrap_or_default()),
    )
        .into_response()
}


#[cfg(test)]
mod answer_deadline_tests {
    use super::*;





}

#[cfg(test)]
mod answer_stream_tests {
    use super::*;
    use futures_util::StreamExt;


    #[tokio::test]
    async fn answer_stream_drop_cancels_owned_work_and_prevents_next_stage() {
        let started = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let later = Arc::new(std::sync::atomic::AtomicBool::new(false));
        struct Dropped(Arc<std::sync::atomic::AtomicBool>);
        impl Drop for Dropped {
            fn drop(&mut self) {
                self.0.store(true, std::sync::atomic::Ordering::SeqCst);
            }
        }
        let execution = Arc::new(crate::answer_execution::AnswerExecution::new(
            Default::default(),
            &ExecutionBudget::default(),
        ));
        let rx = execution.subscribe();
        let work = {
            let started = started.clone();
            let dropped = dropped.clone();
            let later = later.clone();
            async move {
                let _guard = Dropped(dropped);
                started.store(true, std::sync::atomic::Ordering::SeqCst);
                std::future::pending::<()>().await;
                later.store(true, std::sync::atomic::Ordering::SeqCst);
                StatusCode::OK.into_response()
            }
        };
        let mut stream = Box::pin(answer_event_stream(work, rx, execution));
        assert!(stream.next().await.is_some()); // Actual admission stage.
        assert!(
            tokio::time::timeout(std::time::Duration::from_millis(10), stream.next())
                .await
                .is_err()
        );
        assert!(started.load(std::sync::atomic::Ordering::SeqCst));
        drop(stream);
        assert!(dropped.load(std::sync::atomic::Ordering::SeqCst));
        assert!(!later.load(std::sync::atomic::Ordering::SeqCst));
    }


}
