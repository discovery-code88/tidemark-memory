use super::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphExportRequest {
    pub conversation_id: Option<String>,
    pub include_procedures: Option<bool>,
    pub include_conflicts: Option<bool>,
    pub limit: Option<usize>,
    pub include_layout: Option<bool>,
    pub layout_mode: Option<String>,
    pub layout_width: Option<f32>,
    pub layout_height: Option<f32>,
    pub viewport_min_x: Option<f32>,
    pub viewport_min_y: Option<f32>,
    pub viewport_max_x: Option<f32>,
    pub viewport_max_y: Option<f32>,
    pub node_offset: Option<usize>,
    pub node_limit: Option<usize>,
    pub edge_offset: Option<usize>,
    pub edge_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphQueryRequest {
    pub conversation_id: Option<String>,
    pub entity_id: Option<String>,
    pub node_kind: Option<String>,
    pub max_depth: Option<usize>,
    pub traversal_direction: Option<String>,
    pub relation_kind: Option<String>,
    pub layer: Option<String>,
    pub conflict_state: Option<String>,
    pub valid_at_after: Option<DateTime<Utc>>,
    pub valid_at_before: Option<DateTime<Utc>>,
    pub invalid_at_after: Option<DateTime<Utc>>,
    pub invalid_at_before: Option<DateTime<Utc>>,
    pub include_procedures: Option<bool>,
    pub include_conflicts: Option<bool>,
    pub limit: Option<usize>,
    pub include_layout: Option<bool>,
    pub layout_mode: Option<String>,
    pub layout_width: Option<f32>,
    pub layout_height: Option<f32>,
    pub viewport_min_x: Option<f32>,
    pub viewport_min_y: Option<f32>,
    pub viewport_max_x: Option<f32>,
    pub viewport_max_y: Option<f32>,
    pub node_offset: Option<usize>,
    pub node_limit: Option<usize>,
    pub edge_offset: Option<usize>,
    pub edge_limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryGraphNode {
    pub id: String,
    pub kind: String,
    pub label: String,
    pub conversation_id: Option<String>,
    pub layer: Option<String>,
    pub status: Option<String>,
    pub valid_at: Option<DateTime<Utc>>,
    pub invalid_at: Option<DateTime<Utc>>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryGraphEdge {
    pub from: String,
    pub to: String,
    pub kind: String,
    pub label: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphLayoutPosition {
    pub node_id: String,
    pub x: f32,
    pub y: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphSpatialIndexCell {
    pub cell_id: String,
    pub min_x: f32,
    pub min_y: f32,
    pub max_x: f32,
    pub max_y: f32,
    pub node_ids: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphLayout {
    pub layout_version: String,
    pub cache_key: String,
    pub mode: String,
    pub width: f32,
    pub height: f32,
    pub spatial_index_cell_size: f32,
    pub spatial_index_cells: Vec<MemoryGraphSpatialIndexCell>,
    pub positions: Vec<MemoryGraphLayoutPosition>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryGraphOperationAudit {
    pub event_id: String,
    pub action: String,
    pub generated_at: DateTime<Utc>,
    pub layout_version: Option<String>,
    pub visible_node_count: usize,
    pub visible_edge_count: usize,
    pub total_node_count: usize,
    pub total_edge_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryGraphAuditListRequest {
    pub limit: Option<usize>,
    pub action: Option<String>,
    pub layout_version: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryGraphAuditListResponse {
    pub items: Vec<MemoryGraphOperationAudit>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryGraphExportResponse {
    pub generated_at: DateTime<Utc>,
    pub conversation_id: Option<String>,
    pub node_count: usize,
    pub edge_count: usize,
    pub total_node_count: usize,
    pub total_edge_count: usize,
    pub node_offset: usize,
    pub edge_offset: usize,
    pub node_limit: usize,
    pub edge_limit: usize,
    pub has_more_nodes: bool,
    pub has_more_edges: bool,
    pub layout: Option<MemoryGraphLayout>,
    pub audit: MemoryGraphOperationAudit,
    pub context_block: String,
    pub nodes: Vec<MemoryGraphNode>,
    pub edges: Vec<MemoryGraphEdge>,
}
