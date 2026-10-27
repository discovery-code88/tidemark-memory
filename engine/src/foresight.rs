use crate::api_types::{RetrievedItem, SemanticAnswerNote};
use crate::MemoryLayer;
use std::collections::HashSet;

pub fn build_foresight_answer_notes(
    items: &[RetrievedItem],
    query: &str,
) -> Vec<SemanticAnswerNote> {
    let query = query.to_lowercase();
    let wants_future = query.contains("接下来")
        || query.contains("接着")
        || query.contains("计划")
        || query.contains("打算")
        || query.contains("明天")
        || query.contains("下周")
        || query.contains("本周")
        || query.contains("截止")
        || query.contains("答应");
    if !wants_future {
        return Vec::new();
    }

    let mut notes = Vec::new();
    let mut seen = HashSet::new();
    for item in items {
        if item.layer != MemoryLayer::Foresight {
            continue;
        }
        let message = format!("[未来承诺提示] 已记录的后续事项：{}。", item.content);
        if seen.insert(message.clone()) {
            notes.push(SemanticAnswerNote {
                note_type: "foresight".to_string(),
                message,
                entity_id: None,
                attribute_key: None,
                scope: None,
                candidate_values: Vec::new(),
            });
        }
    }

    notes
}
