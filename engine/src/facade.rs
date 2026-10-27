//! Public entry points of the memory engine: store memories, search them.
//!
//! The engine keeps memories in memory; a host that needs durability keeps
//! its own write log and replays it through [`Engine::ingest`] on start.

use crate::*;
use chrono::{DateTime, Utc};

/// One ranked search result.
#[derive(Debug, Clone)]
pub struct Hit {
    pub id: u64,
    pub content: String,
    pub score: f32,
    pub created_at: DateTime<Utc>,
}

/// Outcome of storing one memory.
#[derive(Debug, Clone)]
pub struct Stored {
    pub accepted: bool,
    pub memory_id: Option<u64>,
    pub reason: String,
}

pub struct Engine {
    inner: MemoryEngine,
}

impl Default for Engine {
    fn default() -> Self {
        Self::new()
    }
}

impl Engine {
    pub fn new() -> Self {
        Self { inner: MemoryEngine::with_write_gate_thresholds(WriteGateThresholds::default()) }
    }

    /// Stores `content` as an event memory of `conversation_id`
    /// (the isolation scope). `source` labels where it came from.
    pub fn ingest(&mut self, conversation_id: &str, content: &str, source: &str) -> Stored {
        let req: IngestRequest = serde_json::from_value(serde_json::json!({
            "conversation_id": conversation_id,
            "layer": "event",
            "content": content,
            "source": source,
        }))
        .expect("ingest request shape");
        let resp = self.inner.ingest(req);
        Stored { accepted: resp.accepted, memory_id: resp.memory_id, reason: resp.reason }
    }

    /// Up to `top_k` memories of `conversation_id` ranked for `query`.
    pub fn search(&mut self, conversation_id: &str, query: &str, top_k: usize) -> Vec<Hit> {
        let req: QueryRequest = serde_json::from_value(serde_json::json!({
            "conversation_id": conversation_id,
            "query": query,
            "top_k": top_k,
        }))
        .expect("query request shape");
        self.inner
            .retrieve(&req)
            .items
            .into_iter()
            .take(top_k)
            .map(|item| Hit { id: item.id, content: item.content, score: item.score, created_at: item.created_at })
            .collect()
    }

    /// Number of stored memories.
    pub fn len(&self) -> usize {
        self.inner.memories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.memories.is_empty()
    }
}

#[cfg(test)]
mod facade_tests {
    use super::*;

    #[test]
    fn stored_messages_are_found_by_search_within_their_scope() {
        let mut engine = Engine::new();
        let messages = [
            "Caroline: I went to a LGBTQ support group yesterday and it was so powerful.",
            "Melanie: That's great! I painted a sunrise over the lake last week.",
            "Caroline: Thanks! I'm also looking into counseling as a career.",
            "Melanie: Thanks!",
        ];
        for m in messages {
            let stored = engine.ingest("conv-a", m, "test");
            assert!(stored.accepted, "{m}: {}", stored.reason);
        }
        engine.ingest("conv-b", "Dave: I fixed a Ford Mustang in the junkyard.", "test");
        let hits = engine.search("conv-a", "What career is Caroline considering?", 3);
        assert!(hits.first().is_some_and(|h| h.content.contains("counseling")), "{hits:?}");
        assert!(engine.search("conv-a", "Ford Mustang", 10).iter().all(|h| !h.content.contains("Mustang")), "scopes are isolated");
    }
}
