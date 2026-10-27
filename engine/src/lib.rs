//! Tidemark memory engine: the memory core of Unlimited Memory (ingestion,
//! retrieval, scoring, dates, answering, profiles), without the server,
//! persistence and advanced modules.
#![allow(dead_code, unused_imports)]

pub mod answer_execution;
pub mod answer_pipeline;
pub mod answer_reasoning_contract;
pub mod answer_reasoning_provider;
pub mod api_types;
pub mod context_support;
pub mod embedding;
pub mod foresight;
pub mod semantic_support;
pub mod memory_engine_semantic_core;
pub mod cognitive;
pub mod llm_protocol;
pub mod memory_engine;
pub mod memory_engine_procedure;
pub mod meta_memory;
pub mod memory_engine_ingest;
pub mod memory_engine_query;
pub mod memory_engine_retrieval;
pub mod memory_engine_summary;
pub mod memory_models;
pub mod memory_quality;
pub mod person_profile;
pub mod provider_egress;
pub mod semantic;
pub mod semantic_handlers;
pub mod semantic_retrieval;
pub mod semantic_scorer;
pub mod shared_object;
pub mod support;

pub(crate) use api_types::CollaborationMessage;
pub(crate) use memory_engine::{
    AdminPrincipalState,
    AdminRoleBindingState, AgentRuntimeState, AutoConsolidationConfig, ConnectorScheduleState,
    ControlPlanePolicyAssignmentState, MemoryEngine, MemoryVisibility, QueryRuntimeSnapshot,
    ScopedQueryContext, WorkspaceRegistryState,
};
pub(crate) use memory_models::{
    ActivationMetadata, CacheEntry, EpisodicMetadata, EventMetadata, MemoryLayer, MemoryRecord,
    ProcedureExecutionResult, ProcedureExecutionState, ProcedureGovernance, ProcedureSchema,
};
pub(crate) use memory_quality::{
    apply_memory_quality_metrics, clean_memory_type, clean_quality_refs, clean_quality_score,
    collect_embedding_stats, collect_layer_stats, memory_layer_label,
    memory_quality_datetime_value, memory_quality_number, memory_quality_score_boost,
    memory_quality_text_value, memory_quality_tuning, memory_quality_type, normalize_memory_type,
    parse_structured_memory_content, set_quality_line,
};
pub(crate) use semantic_handlers::answer_memory;
pub(crate) use semantic_handlers::spawn_semantic_sleep_worker;
use api_types::{
    api_error, ActivationPromoteRequest, ActivationPromoteResponse, ActivationQueryRequest,
    ActivationQueryResponse, ActivationUpdateRequest, ActivationUpdateResponse,
    AdminAuditApplicationSummary, AdminAuditExportRequest, AdminAuditExportResponse,
    AdminBackupPostureRequest, AdminBackupPostureResponse, AdminComplianceReportRequest,
    AdminComplianceReportResponse, AdminOperatorReportRequest, AdminOperatorReportResponse,
    AdminPrincipalBatchLifecycleRequest, AdminPrincipalBatchLifecycleResponse,
    AdminPrincipalInspectRequest, AdminPrincipalInspectResponse, AdminPrincipalLifecycleRequest,
    AdminPrincipalLifecycleResponse, AdminPrincipalListRequest, AdminPrincipalListResponse,
    AdminPrincipalRecord, AdminPrincipalUpsertRequest, AdminPrincipalUpsertResponse,
    AdminRoleBindingBatchLifecycleRequest, AdminRoleBindingBatchLifecycleResponse,
    AdminRoleBindingInspectRequest, AdminRoleBindingInspectResponse,
    AdminRoleBindingLifecycleRequest, AdminRoleBindingLifecycleResponse,
    AdminRoleBindingListRequest, AdminRoleBindingListResponse, AdminRoleBindingRecord,
    AdminRoleBindingUpsertRequest, AdminRoleBindingUpsertResponse, AdminSecurityPostureRequest,
    AdminSecurityPostureResponse, AgentMessageInboxRequest, AgentMessageInboxResponse,
    AgentMessageSendRequest, AgentMessageSendResponse, AgentRuntimeControlRequest,
    AgentRuntimeControlResponse, AgentRuntimeInspectRequest, AgentRuntimeInspectResponse,
    AgentRuntimeListRequest, AgentRuntimeListResponse, AgentRuntimeSummary,
    AgentRuntimeUpsertRequest, AgentRuntimeUpsertResponse, AgentScopedIngestRequest,
    AgentScopedQueryRequest, AgentTaskDispatchRequest, AgentTaskDispatchResponse,
    AgentTaskListRequest, AgentTaskListResponse, AgentTaskRecord, AgentTaskUpdateRequest,
    AgentTaskUpdateResponse, AnswerRequest, AnswerResponse, BackendResponse,
    BackupRestorePostureSummary, BackupRestoreWorkspacePosture, CadenceSettingsUpsertRequest,
    CanonicalSemanticAuditEntry, CanonicalSemanticItem, CanonicalSemanticMutationRequest,
    CanonicalSemanticMutationResponse, CanonicalSemanticQueryRequest, CanonicalSemanticResponse,
    ClusteringResult, CollaborationCompactPackageActionRequest,
    CollaborationCompactPackageActionResponse, CollaborationCompactPackageItem,
    CollaborationCompactPackagesRequest, CollaborationCompactPackagesResponse,
    CollaborationCompactRequest, CollaborationCompactResponse, CollaborationCompactedOperation,
    CollaborationEvent, CollaborationMergePlanRequest, CollaborationOfflineBranchActionRequest,
    CollaborationOfflineBranchActionResponse, CollaborationOfflineBranchCreateRequest,
    CollaborationOfflineBranchItem, CollaborationOfflineBranchesRequest,
    CollaborationOfflineBranchesResponse, ComplianceViolationCounts, ComplianceWorkspaceReportItem,
    ConnectorCatalogSummary, ConnectorOperationsFindingsSummary, ConnectorOperationsReportRequest,
    ConnectorOperationsReportResponse, ConnectorPreviewItem, ConnectorPullRequest,
    ConnectorPullResponse, ConnectorRetryPostureSummary, ConnectorRunRecord, ConnectorRunsRequest,
    ConnectorRunsResponse, ConnectorRunsSummary, ConnectorScheduleRecord,
    ConnectorScheduleRunRequest, ConnectorScheduleRunResponse, ConnectorScheduleUpsertRequest,
    ConnectorScheduleUpsertResponse, ConnectorSchedulesRequest, ConnectorSchedulesResponse,
    ConnectorSchedulesSummary, ConnectorStatusItem, ConnectorStatusRequest,
    ConnectorStatusResponse, ConnectorWebhookSecretUpsertRequest,
    ConnectorWebhookSecretUpsertResponse, ConsolidationClusterSummary, ConsolidationJob,
    ConsolidationJobMetrics, ConsolidationJobsResponse, ConsolidationRunRequest,
    ContainerAuditEntry, ControlPlaneActionPermissionSummary, ControlPlaneAuditEntry,
    ControlPlaneAuditSummary, ControlPlaneLifecycleSummary, ControlPlaneOperatorSummary,
    ControlPlanePolicyAssignmentBatchLifecycleRequest,
    ControlPlanePolicyAssignmentBatchLifecycleResponse, ControlPlanePolicyAssignmentInspectRequest,
    ControlPlanePolicyAssignmentInspectResponse, ControlPlanePolicyAssignmentLifecycleRequest,
    ControlPlanePolicyAssignmentLifecycleResponse, ControlPlanePolicyAssignmentListRequest,
    ControlPlanePolicyAssignmentListResponse, ControlPlanePolicyAssignmentRecord,
    ControlPlanePolicyAssignmentUpsertRequest, ControlPlanePolicyAssignmentUpsertResponse,
    ControlPlaneRolePermissionSummary, CoverageReport, CycleResult, DecayResponse,
    GatewayRuntimeClientRequirement, GatewayRuntimeConformanceSummary,
    GatewayRuntimeEntrypointStatus, GatewayRuntimeEnvelope, GatewayRuntimeStatusRequest,
    GatewayRuntimeStatusResponse, GatewayRuntimeStepStatus, GatewayRuntimeThirdPartyGuard,
    GovernanceAlert, GovernanceAlertsResponse,
    GovernanceConflictViewResponse, GovernanceConnectorActivitySummary,
    GovernanceConnectorRetryPostureSummary, GovernanceConnectorSecuritySummary,
    GovernanceContainerActivitySummary, GovernanceDashboardRequest, GovernanceDashboardResponse,
    GovernanceJobViewResponse, GovernanceJobWindowSummary, GovernanceMultimodalCounts,
    GovernanceSnapshotResponse, IngestRequest, IngestResponse, IntelligenceLayerStatus,
    IntelligenceStatusResponse, MemoryAutoOrganizeRequest, MemoryAutoOrganizeResponse,
    MemoryBatchUpdateRequest, MemoryBatchUpdateResponse, MemoryBrowserItem,
    MemoryCapsuleContextPackage, MemoryContainerBundle, MemoryContainerExportRequest,
    MemoryContainerExportResponse, MemoryContainerImportRequest, MemoryContainerImportResponse,
    MemoryContextAgentBootstrapRequest, MemoryContextAgentBootstrapResponse,
    MemoryContextAgentClientOperation, MemoryContextAgentDisplayArtifact,
    MemoryContextAgentExecutionContract, MemoryContextAgentNextAction,
    MemoryContextAgentTraceContract, MemoryContextAgentTurnStep, MemoryContextBatchRollbackRequest,
    MemoryContextBatchRollbackResponse, MemoryContextBuildRequest, MemoryContextBuildResponse,
    MemoryContextCommitRequest, MemoryContextCommitResponse, MemoryContextDecideRequest,
    MemoryContextDecideResponse, MemoryContextDurableWriteGate, MemoryContextEvidence,
    MemoryContextManagedTurnContinueRequest, MemoryContextManagedTurnRequest,
    MemoryContextManagedTurnResponse, MemoryContextMergeBatchFailure,
    MemoryContextMergeBatchRequest, MemoryContextMergeBatchResponse, MemoryContextMergeDiffItem,
    MemoryContextMergeRequest, MemoryContextMergeResponse, MemoryContextMergeRiskAssessment,
    MemoryContextMergeSuggestion, MemoryContextPolicyAuditRequest,
    MemoryContextPolicyAuditResponse, MemoryContextPolicyDecision, MemoryContextProjectRequest,
    MemoryContextProjectResponse, MemoryContextProposeRequest, MemoryContextProposeResponse,
    MemoryContextResumeRequest, MemoryContextResumeResponse, MemoryContextReviewActionRequest,
    MemoryContextReviewActionResponse, MemoryContextReviewItem, MemoryContextReviewRequest,
    MemoryContextReviewResponse, MemoryContextRollbackRequest, MemoryContextRollbackResponse,
    MemoryContextTasteArbitrationCheckRequest, MemoryContextTasteArbitrationCheckResponse,
    MemoryContextTasteGraphAlgorithmsRequest, MemoryContextTasteGraphAlgorithmsResponse,
    MemoryContextTasteGraphCorrectionRequest, MemoryContextTasteGraphCorrectionResponse,
    MemoryContextTasteGraphEditRequest, MemoryContextTasteGraphEditResponse,
    MemoryContextTasteGraphLayoutLoadRequest, MemoryContextTasteGraphLayoutLoadResponse,
    MemoryContextTasteGraphLayoutSaveRequest, MemoryContextTasteGraphLayoutSaveResponse,
    MemoryContextTasteGraphProvenanceEntry, MemoryContextTasteGraphProvenanceRequest,
    MemoryContextTasteGraphProvenanceResponse, MemoryContextTasteGraphTraverseRequest,
    MemoryContextTasteGraphTraverseResponse, MemoryContextTasteProfileCluster,
    MemoryContextTasteProfileNode, MemoryContextTasteProfileRequest,
    MemoryContextTasteProfileResponse, MemoryContextTurnContractValidateRequest,
    MemoryContextTurnContractValidateResponse, MemoryContextTurnContractValidationCheck,
    MemoryContextTurnReceiptItem, MemoryContextTurnReceiptRequest,
    MemoryContextTurnReceiptResponse, MemoryContextTurnReceiptsRequest,
    MemoryContextTurnReceiptsResponse, MemoryContextWriteCandidate, MemoryDeleteRequest,
    MemoryDeleteResponse, MemoryFactVersionItem, MemoryFileDiffHunk, MemoryFileDiffLine,
    MemoryFileDiffRow, MemoryFileEditConflict, MemoryFileEvidence, MemoryFileListRequest,
    MemoryFileListResponse, MemoryFileMergeConflict, MemoryFileReadRequest, MemoryFileReadResponse,
    MemoryFileRecord, MemoryFileRollbackConflict, MemoryFileRollbackPreviewRequest,
    MemoryFileRollbackPreviewResponse, MemoryFileRollbackRequest, MemoryFileRollbackResponse,
    MemoryFileSearchRequest, MemoryFileSearchResponse, MemoryFileSectionDiff,
    MemoryFileSectionEditRequest, MemoryFileSectionEditResponse, MemoryFileSectionMergeRequest,
    MemoryFileSectionMergeResponse, MemoryFileVisualDiff, MemoryFileWriteRequest,
    MemoryFileWriteResponse, MemoryFolderLlmNameRequest, MemoryFolderLlmNameResponse,
    MemoryFolderSuggestion, MemoryFolderSummary, MemoryGraphAuditListRequest,
    MemoryGraphAuditListResponse, MemoryGraphEdge, MemoryGraphExportRequest,
    MemoryGraphExportResponse, MemoryGraphLayout, MemoryGraphLayoutPosition, MemoryGraphNode,
    MemoryGraphOperationAudit, MemoryGraphQueryRequest, MemoryGraphSpatialIndexCell,
    MemoryInterchangeBundle, MemoryInterchangeCapabilities, MemoryInterchangeCapabilitiesResponse,
    MemoryInterchangeCompatibility, MemoryInterchangeDeviceProfile, MemoryInterchangeExportRequest,
    MemoryInterchangeForeignMergePolicy, MemoryInterchangeImportRequest,
    MemoryInterchangeIntegrity, MemoryInterchangeModelProfile, MemoryInterchangeNegotiationRequest,
    MemoryInterchangeNegotiationResponse, MemoryInterchangePortability, MemoryInterchangeProducer,
    MemoryInterchangeValidateRequest, MemoryInterchangeValidateResponse, MemoryListRequest,
    MemoryListResponse, MemoryRecallPolicyEnvelope, MemoryRecallRequest, MemoryReflectRequest,
    MemoryRetainRequest, MemorySpaceListRequest, MemorySpaceListResponse, MemorySpaceRecord,
    MemoryTraceItem, MemoryTraceResponse, MemoryUpdateRequest, MemoryUpdateResponse,
    MetaMemoryEvaluateResponse, MetaMemoryEvaluatedItem, MetricsResponse,
    MultimodalBatchGovernance, MultimodalBatchIngestRequest, MultimodalBatchIngestResponse,
    MultimodalDerivedItem, MultimodalExtractRequest, MultimodalExtractResponse,
    MultimodalExtractionRunRecord, MultimodalExtractionRunsRequest,
    MultimodalExtractionRunsResponse, MultimodalIngestRequest, MultimodalIngestResponse,
    MultimodalKind, NovelImportExtractRequest, NovelImportExtractResponse,
    NovelImportJobActionRequest, NovelImportJobListRequest, NovelImportJobListResponse,
    NovelImportJobRecord, NovelImportJobSubmitResponse, NovelUploadChunkRequest,
    NovelUploadChunkResponse, NovelUploadFinishRequest, NovelUploadStartRequest,
    NovelUploadStartResponse, OcrUploadResponse, OperatorFindingSummary, PageRankResult,  PolicyViolationSummary, PortableMemoryRecord,
    ProcedureActivateRequest, ProcedureActivationResponse, ProcedureConsolidateRequest,
    ProcedureConsolidateResponse, ProcedureEvaluateRequest, ProcedureEvaluateResponse,
    ProcedureExecutionRequest, ProcedureExecutionResponse, ProcedureImportRollbackRequest,
    ProcedureImportRollbackResponse, ProcedureImportSkillFileRequest,
    ProcedureImportSkillFileResponse, ProcedureStatusRequest, ProcedureStatusResponse,
    ProcedureStepStatus, ProcedureSyncSkillFileItem, ProcedureSyncSkillFilesRequest,
    ProcedureSyncSkillFilesResponse, ProvenanceEdge, ProvenanceEdgeCreateRequest,
    ProvenanceEdgeUpsertResponse, ProvenanceGraphQueryRequest, ProvenanceGraphQueryResponse,
    ProvenanceNode, ProvenanceNodeCreateRequest, ProvenanceNodeUpsertResponse, PruneResponse,
    QueryRequest, QueryResponse, ReindexRequest, ReindexResponse, RetrievedItem,
    SecurityPostureSummary, SecurityWorkspacePosture, SemanticAnswerNote,
    SemanticSleepCycleResponse, SummaryResponse, TenantRegistryInspectRequest,
    TenantRegistryInspectResponse, TenantRegistryListRequest, TenantRegistryListResponse,
    TenantRegistryRecord, TypedRollbackRequest, TypedRollbackResponse,
    WorkspaceAdminSummaryRequest, WorkspaceAdminSummaryResponse, WorkspaceConversationSummary,
    WorkspaceRegistryListRequest, WorkspaceRegistryListResponse, WorkspaceRegistryRecord,
    WorkspaceRegistryUpsertRequest, WorkspaceRegistryUpsertResponse, WorkspaceSecuritySummary,
};
use axum::{
    body::{to_bytes, Body, Bytes},
    extract::{ConnectInfo, DefaultBodyLimit, Extension, Multipart, Path, State},
    http::{header, HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{Html, IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use chrono::{DateTime, Duration, Utc};
use embedding::EmbeddingClient;
use hmac::{Hmac, Mac};
use semantic::{
    apply_conflict_resolution, build_semantic_conflict_set, infer_semantic_scope,
    maybe_build_semantic_node, merge_semantic_confirmation,
    semantic_applicability_score_adjustment, semantic_kind_score_adjustment,
    semantic_scope_allows_query, semantic_scope_label, semantic_scope_score_adjustment,
    semantic_status_score_adjustment, try_resolve_conflict, SemanticConflictSet, SemanticNode,
    SemanticStatus,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet, VecDeque},
    convert::Infallible,
    env, fs,
    io::Read,
    net::SocketAddr,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering as AtomicOrdering},
        Arc, OnceLock,
    },
    time::{Duration as StdDuration, Instant},
};
use tokio::sync::{Mutex, Notify, RwLock};

pub use support::*;
pub(crate) use cognitive::*;
pub(crate) use meta_memory::*;
pub(crate) use context_support::*;
pub(crate) use semantic_support::*;
pub(crate) use foresight::*;

pub mod facade;
pub use facade::{Engine, Hit, Stored};
