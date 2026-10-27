use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceAdminSummaryRequest {
    pub limit: Option<usize>,
    pub workspace_id: Option<String>,
    pub tenant_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceSecuritySummary {
    pub auth_enabled: bool,
    pub memory_api_key_count: usize,
    pub admin_api_key_count: usize,
    pub max_body_bytes: usize,
    pub request_timeout_ms: u64,
    pub connector_webhook_protected_conversations: usize,
    pub connector_replay_protection_enabled: bool,
    pub connector_timestamp_skew_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceConversationSummary {
    pub conversation_id: String,
    pub memory_count: usize,
    pub last_memory_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PolicyViolationSummary {
    pub code: String,
    pub severity: String,
    pub tenant_id: String,
    pub workspace_id: String,
    pub affected_conversation_count: usize,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlaneAuditSummary {
    pub total_events: usize,
    pub applied_events: usize,
    pub denied_events: usize,
    pub principal_change_events: usize,
    pub policy_change_events: usize,
    pub last_recorded_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceAdminSummaryResponse {
    pub generated_at: DateTime<Utc>,
    pub workspace_id: String,
    pub tenant_id: String,
    pub backend: String,
    pub embedding: String,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub conversation_count: usize,
    pub memory_count: usize,
    pub layer_counts: HashMap<String, usize>,
    pub multimodal_counts: GovernanceMultimodalCounts,
    pub high_priority_job_count: usize,
    pub disputed_semantic_count: usize,
    pub policy_assignments: Vec<ControlPlanePolicyAssignmentRecord>,
    pub connector_run_count: usize,
    pub connector_schedule_count: usize,
    pub policy_violations: Vec<PolicyViolationSummary>,
    pub control_plane_audit_summary: ControlPlaneAuditSummary,
    pub control_plane_operator_summary: ControlPlaneOperatorSummary,
    pub security: WorkspaceSecuritySummary,
    pub metrics: MetricsResponse,
    pub recent_conversations: Vec<WorkspaceConversationSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkspaceRegistryUpsertRequest {
    pub workspace_id: String,
    pub tenant_id: Option<String>,
    pub display_name: Option<String>,
    pub isolation_policy: Option<String>,
    pub compliance_profile: Option<String>,
    pub conversation_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceRegistryRecord {
    pub workspace_id: String,
    pub tenant_id: String,
    pub display_name: Option<String>,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub lifecycle_status: String,
    pub conversation_count: usize,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceRegistryUpsertResponse {
    pub workspace: WorkspaceRegistryRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorkspaceRegistryListRequest {
    pub limit: Option<usize>,
    pub tenant_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceRegistryListResponse {
    pub items: Vec<WorkspaceRegistryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemorySpaceListRequest {
    pub limit: Option<usize>,
    pub tenant_id: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySpaceRecord {
    pub space_id: String,
    pub tenant_id: String,
    pub display_name: Option<String>,
    pub mission: String,
    pub directives: Vec<String>,
    pub traits: Vec<String>,
    pub conversation_count: usize,
    pub memory_count: usize,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemorySpaceListResponse {
    pub items: Vec<MemorySpaceRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TenantRegistryRecord {
    pub tenant_id: String,
    pub workspace_count: usize,
    pub conversation_count: usize,
    pub memory_count: usize,
    pub isolation_policies: Vec<String>,
    pub compliance_profiles: Vec<String>,
    pub updated_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TenantRegistryListRequest {
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TenantRegistryListResponse {
    pub items: Vec<TenantRegistryRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantRegistryInspectRequest {
    pub tenant_id: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TenantRegistryInspectResponse {
    pub tenant: TenantRegistryRecord,
    pub workspaces: Vec<WorkspaceRegistryRecord>,
    pub policy_assignments: Vec<ControlPlanePolicyAssignmentRecord>,
    pub policy_violations: Vec<PolicyViolationSummary>,
    pub control_plane_audit_summary: ControlPlaneAuditSummary,
    pub control_plane_operator_summary: ControlPlaneOperatorSummary,
    pub recent_conversations: Vec<WorkspaceConversationSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminComplianceReportRequest {
    pub workspace_id: Option<String>,
    pub tenant_id: Option<String>,
    pub limit: Option<usize>,
    pub control_plane_outcome: Option<String>,
    pub finding_severity: Option<String>,
    pub finding_code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceViolationCounts {
    pub total: usize,
    pub high: usize,
    pub medium: usize,
    pub low: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComplianceWorkspaceReportItem {
    pub workspace_id: String,
    pub tenant_id: String,
    pub display_name: Option<String>,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub conversation_count: usize,
    pub policy_assignment_count: usize,
    pub policy_violation_count: usize,
    pub denied_event_count: usize,
    pub last_control_plane_event_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminComplianceReportResponse {
    pub generated_at: DateTime<Utc>,
    pub workspace_id: String,
    pub tenant_id: String,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub workspaces: Vec<ComplianceWorkspaceReportItem>,
    pub policy_assignments: Vec<ControlPlanePolicyAssignmentRecord>,
    pub policy_violations: Vec<PolicyViolationSummary>,
    pub violation_counts: ComplianceViolationCounts,
    pub control_plane_audit_summary: ControlPlaneAuditSummary,
    pub control_plane_operator_summary: ControlPlaneOperatorSummary,
    pub control_plane_lifecycle_summary: ControlPlaneLifecycleSummary,
    pub control_plane_permission_matrix: Vec<ControlPlaneRolePermissionSummary>,
    pub control_plane_action_permissions: Vec<ControlPlaneActionPermissionSummary>,
    pub backup_restore_summary: BackupRestorePostureSummary,
    pub backup_restore_workspaces: Vec<BackupRestoreWorkspacePosture>,
    pub backup_restore_findings: Vec<OperatorFindingSummary>,
    pub security_posture_summary: SecurityPostureSummary,
    pub security_posture_workspaces: Vec<SecurityWorkspacePosture>,
    pub security_posture_findings: Vec<OperatorFindingSummary>,
    pub findings: Vec<OperatorFindingSummary>,
    pub recent_control_plane_events: Vec<ControlPlaneAuditEntry>,
    pub recent_principal_change_events: Vec<ControlPlaneAuditEntry>,
    pub recent_policy_change_events: Vec<ControlPlaneAuditEntry>,
    pub security: WorkspaceSecuritySummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminOperatorReportRequest {
    pub workspace_id: Option<String>,
    pub tenant_id: Option<String>,
    pub limit: Option<usize>,
    pub finding_severity: Option<String>,
    pub finding_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminBackupPostureRequest {
    pub workspace_id: Option<String>,
    pub tenant_id: Option<String>,
    pub limit: Option<usize>,
    pub finding_severity: Option<String>,
    pub finding_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminSecurityPostureRequest {
    pub workspace_id: Option<String>,
    pub tenant_id: Option<String>,
    pub limit: Option<usize>,
    pub finding_severity: Option<String>,
    pub finding_code: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OperatorFindingSummary {
    pub code: String,
    pub severity: String,
    pub tenant_id: Option<String>,
    pub workspace_id: Option<String>,
    pub principal_id: Option<String>,
    pub binding_id: Option<String>,
    pub assignment_id: Option<String>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackupRestorePostureSummary {
    pub status: String,
    pub total_workspaces: usize,
    pub configured_workspaces: usize,
    pub backup_required_workspaces: usize,
    pub disabled_workspaces: usize,
    pub missing_provider_workspaces: usize,
    pub missing_recent_backup_workspaces: usize,
    pub stale_backup_workspaces: usize,
    pub missing_restore_drill_workspaces: usize,
    pub stale_restore_drill_workspaces: usize,
    pub missing_retention_workspaces: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackupRestoreWorkspacePosture {
    pub workspace_id: String,
    pub tenant_id: String,
    pub compliance_profile: String,
    pub backup_policy: String,
    pub backup_provider: Option<String>,
    pub backup_last_success_at: Option<DateTime<Utc>>,
    pub restore_drill_last_success_at: Option<DateTime<Utc>>,
    pub backup_retention_days: Option<u64>,
    pub status: String,
    pub finding_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminBackupPostureResponse {
    pub generated_at: DateTime<Utc>,
    pub workspace_id: String,
    pub tenant_id: String,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub summary: BackupRestorePostureSummary,
    pub workspaces: Vec<BackupRestoreWorkspacePosture>,
    pub findings: Vec<OperatorFindingSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SecurityPostureSummary {
    pub status: String,
    pub total_workspaces: usize,
    pub auth_enabled: bool,
    pub configured_workspaces: usize,
    pub missing_tls_policy_workspaces: usize,
    pub tls_disabled_workspaces: usize,
    pub missing_external_secrets_provider_workspaces: usize,
    pub missing_secret_rotation_workspaces: usize,
    pub stale_secret_rotation_workspaces: usize,
    pub missing_audit_retention_workspaces: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SecurityWorkspacePosture {
    pub workspace_id: String,
    pub tenant_id: String,
    pub compliance_profile: String,
    pub tls_policy: String,
    pub external_secrets_provider_configured: bool,
    pub secret_rotation_last_at: Option<DateTime<Utc>>,
    pub secret_rotation_days: Option<u64>,
    pub audit_log_retention_days: Option<u64>,
    pub status: String,
    pub finding_codes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminSecurityPostureResponse {
    pub generated_at: DateTime<Utc>,
    pub workspace_id: String,
    pub tenant_id: String,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub summary: SecurityPostureSummary,
    pub workspaces: Vec<SecurityWorkspacePosture>,
    pub findings: Vec<OperatorFindingSummary>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminOperatorReportResponse {
    pub generated_at: DateTime<Utc>,
    pub workspace_id: String,
    pub tenant_id: String,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub control_plane_operator_summary: ControlPlaneOperatorSummary,
    pub control_plane_lifecycle_summary: ControlPlaneLifecycleSummary,
    pub control_plane_permission_matrix: Vec<ControlPlaneRolePermissionSummary>,
    pub control_plane_action_permissions: Vec<ControlPlaneActionPermissionSummary>,
    pub backup_restore_summary: BackupRestorePostureSummary,
    pub backup_restore_workspaces: Vec<BackupRestoreWorkspacePosture>,
    pub backup_restore_findings: Vec<OperatorFindingSummary>,
    pub security_posture_summary: SecurityPostureSummary,
    pub security_posture_workspaces: Vec<SecurityWorkspacePosture>,
    pub security_posture_findings: Vec<OperatorFindingSummary>,
    pub principals: Vec<AdminPrincipalRecord>,
    pub role_bindings: Vec<AdminRoleBindingRecord>,
    pub policy_assignments: Vec<ControlPlanePolicyAssignmentRecord>,
    pub recent_lifecycle_events: Vec<ControlPlaneAuditEntry>,
    pub findings: Vec<OperatorFindingSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminAuditExportRequest {
    pub workspace_id: Option<String>,
    pub tenant_id: Option<String>,
    pub conversation_id: Option<String>,
    pub limit: Option<usize>,
    pub control_plane_action: Option<String>,
    pub control_plane_outcome: Option<String>,
    pub control_plane_actor_role: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminAuditApplicationSummary {
    pub application_id: String,
    pub candidate_id: String,
    pub conversation_id: String,
    pub candidate_type: String,
    pub applied_at: DateTime<Utc>,
    pub status: String,
    pub created_memory_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContainerAuditEntry {
    pub operation_id: String,
    pub operation: String,
    pub bundle_version: String,
    pub recorded_at: DateTime<Utc>,
    pub conversation_id: Option<String>,
    pub target_conversation_id: Option<String>,
    pub memory_count: usize,
    pub activation_memory_count: usize,
    pub memory_ids: Vec<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlaneAuditEntry {
    pub event_id: String,
    pub recorded_at: DateTime<Utc>,
    pub actor_key_fingerprint: String,
    pub actor_principal_id: Option<String>,
    pub actor_role: Option<String>,
    pub actor_scope_source: String,
    pub action: String,
    pub resource_type: String,
    pub resource_id: String,
    pub tenant_id: Option<String>,
    pub workspace_id: Option<String>,
    pub conversation_id: Option<String>,
    pub outcome: String,
    pub reason: Option<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminAuditExportResponse {
    pub generated_at: DateTime<Utc>,
    pub workspace_id: String,
    pub tenant_id: String,
    pub isolation_policy: String,
    pub compliance_profile: String,
    pub conversation_id: Option<String>,
    pub policy_assignments: Vec<ControlPlanePolicyAssignmentRecord>,
    pub policy_violations: Vec<PolicyViolationSummary>,
    pub control_plane_audit_summary: ControlPlaneAuditSummary,
    pub control_plane_operator_summary: ControlPlaneOperatorSummary,
    pub security: WorkspaceSecuritySummary,
    pub metrics: MetricsResponse,
    pub canonical_audit_entries: Vec<CanonicalSemanticAuditEntry>,
    pub consolidation_jobs: Vec<ConsolidationJob>,
    pub container_operations: Vec<ContainerAuditEntry>,
    pub control_plane_events: Vec<ControlPlaneAuditEntry>,
    pub connector_runs: Vec<ConnectorRunRecord>,
    pub connector_schedules: Vec<ConnectorScheduleRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlaneOperatorSummary {
    pub total_principals: usize,
    pub active_principals: usize,
    pub inactive_principals: usize,
    pub total_role_bindings: usize,
    pub active_role_bindings: usize,
    pub inactive_role_bindings: usize,
    pub inactive_principals_with_active_bindings: usize,
    pub principal_type_counts: HashMap<String, usize>,
    pub active_role_counts: HashMap<String, usize>,
    pub total_policy_assignments: usize,
    pub active_policy_assignments: usize,
    pub inactive_policy_assignments: usize,
    pub tenant_policy_assignments: usize,
    pub workspace_policy_assignments: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlaneLifecycleSummary {
    pub total_events: usize,
    pub principal_suspend_events: usize,
    pub principal_reactivate_events: usize,
    pub role_binding_suspend_events: usize,
    pub role_binding_reactivate_events: usize,
    pub policy_suspend_events: usize,
    pub policy_reactivate_events: usize,
    pub last_recorded_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlaneRolePermissionSummary {
    pub role: String,
    pub can_view_workspace_scope: bool,
    pub can_view_tenant_scope: bool,
    pub can_read_audit: bool,
    pub can_read_compliance_report: bool,
    pub can_read_operator_report: bool,
    pub can_manage_workspace_registry: bool,
    pub can_manage_connector_admin: bool,
    pub can_manage_principals: bool,
    pub can_manage_role_bindings: bool,
    pub can_manage_policy_assignments: bool,
    pub can_execute_runtime_writes: bool,
    pub can_shutdown: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlaneActionPermissionSummary {
    pub action: String,
    pub category: String,
    pub scope_requirement: String,
    pub allowed_roles: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminPrincipalUpsertRequest {
    pub principal_id: String,
    pub display_name: Option<String>,
    pub api_key: String,
    pub status: Option<String>,
    pub principal_type: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalRecord {
    pub principal_id: String,
    pub display_name: Option<String>,
    pub key_fingerprint: String,
    pub status: String,
    pub principal_type: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalUpsertResponse {
    pub principal: AdminPrincipalRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminPrincipalListRequest {
    pub status: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalListResponse {
    pub items: Vec<AdminPrincipalRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminPrincipalInspectRequest {
    pub principal_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalInspectResponse {
    pub principal: AdminPrincipalRecord,
    pub role_bindings: Vec<AdminRoleBindingRecord>,
    pub recent_lifecycle_events: Vec<ControlPlaneAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminPrincipalLifecycleRequest {
    pub principal_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalLifecycleResponse {
    pub principal: AdminPrincipalRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminPrincipalBatchLifecycleRequest {
    pub principal_ids: Vec<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalBatchLifecycleItem {
    pub principal_id: String,
    pub outcome: String,
    pub error_code: Option<String>,
    pub principal: Option<AdminPrincipalRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminPrincipalBatchLifecycleResponse {
    pub applied: usize,
    pub denied: usize,
    pub not_found: usize,
    pub items: Vec<AdminPrincipalBatchLifecycleItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminRoleBindingUpsertRequest {
    pub binding_id: String,
    pub principal_id: String,
    pub role: String,
    pub status: Option<String>,
    pub tenant_ids: Vec<String>,
    pub workspace_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingRecord {
    pub binding_id: String,
    pub principal_id: String,
    pub role: String,
    pub status: String,
    pub tenant_ids: Vec<String>,
    pub workspace_ids: Vec<String>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingUpsertResponse {
    pub binding: AdminRoleBindingRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdminRoleBindingListRequest {
    pub principal_id: Option<String>,
    pub role: Option<String>,
    pub status: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingListResponse {
    pub items: Vec<AdminRoleBindingRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminRoleBindingInspectRequest {
    pub binding_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingInspectResponse {
    pub binding: AdminRoleBindingRecord,
    pub principal: Option<AdminPrincipalRecord>,
    pub recent_lifecycle_events: Vec<ControlPlaneAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminRoleBindingLifecycleRequest {
    pub binding_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingLifecycleResponse {
    pub binding: AdminRoleBindingRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdminRoleBindingBatchLifecycleRequest {
    pub binding_ids: Vec<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingBatchLifecycleItem {
    pub binding_id: String,
    pub outcome: String,
    pub error_code: Option<String>,
    pub binding: Option<AdminRoleBindingRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AdminRoleBindingBatchLifecycleResponse {
    pub applied: usize,
    pub denied: usize,
    pub not_found: usize,
    pub items: Vec<AdminRoleBindingBatchLifecycleItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlanePolicyAssignmentUpsertRequest {
    pub assignment_id: String,
    pub tenant_id: Option<String>,
    pub workspace_id: Option<String>,
    pub policy_key: String,
    pub policy_value: String,
    pub status: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentRecord {
    pub assignment_id: String,
    pub tenant_id: Option<String>,
    pub workspace_id: Option<String>,
    pub policy_key: String,
    pub policy_value: String,
    pub status: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentUpsertResponse {
    pub assignment: ControlPlanePolicyAssignmentRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ControlPlanePolicyAssignmentListRequest {
    pub tenant_id: Option<String>,
    pub workspace_id: Option<String>,
    pub policy_key: Option<String>,
    pub status: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentListResponse {
    pub items: Vec<ControlPlanePolicyAssignmentRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlanePolicyAssignmentInspectRequest {
    pub assignment_id: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentInspectResponse {
    pub assignment: ControlPlanePolicyAssignmentRecord,
    pub effective: bool,
    pub recent_lifecycle_events: Vec<ControlPlaneAuditEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlanePolicyAssignmentLifecycleRequest {
    pub assignment_id: String,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentLifecycleResponse {
    pub assignment: ControlPlanePolicyAssignmentRecord,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlPlanePolicyAssignmentBatchLifecycleRequest {
    pub assignment_ids: Vec<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentBatchLifecycleItem {
    pub assignment_id: String,
    pub outcome: String,
    pub error_code: Option<String>,
    pub assignment: Option<ControlPlanePolicyAssignmentRecord>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ControlPlanePolicyAssignmentBatchLifecycleResponse {
    pub applied: usize,
    pub denied: usize,
    pub not_found: usize,
    pub items: Vec<ControlPlanePolicyAssignmentBatchLifecycleItem>,
}
