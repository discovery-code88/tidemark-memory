# tidemark-engine

The memory engine of Tidemark: the memory core of Unlimited Memory, released as a Rust library.

## What it does

- **Ingestion.** Each memory (event, fact, procedure, working state …) passes a write gate. The engine detects conflicts and supersessions, and keeps per-memory meta-memory scores: salience, stability, utility and conflict pressure. From the stream of writes it derives episode, self-model and autobiographical memories.
- **Retrieval.** Lexical ranking with query intents (temporal, procedural, preference …), temporal-relation clauses, latest-state handling for facts, and memory layers.
- **Semantic core.** Canonical semantic state per entity and attribute.
- **Procedural memory.** Procedure schemas, activation and governance.
- **Dates.** Relative-date resolution against the session date ("last week", "two years ago").
- **Answering and verification.** Typed answers checked statement by statement against verbatim quotations, plus premise checks, and code-derived dates, intervals and orderings.
- **Person profiles.** Facts about a person collected from a whole conversation, each citing its sentences.

## Use

```rust
use tidemark_engine::Engine;

let mut engine = Engine::new();
engine.ingest("conversation-1", "Caroline: I went to a support group yesterday.", "chat");
for hit in engine.search("conversation-1", "What did Caroline do?", 10) {
    println!("{:.3} {}", hit.score, hit.content);
}
```

The engine keeps memories in memory. A host that needs durability keeps its own write log and replays it through `Engine::ingest` on start; Tidemark does this with SQLite.

Answering, verification and profiles are library code. Tidemark's service exposes only Add and Search.

## Optional local models

The engine can use local ONNX models: an embedding index (`SEMANTIC_RETRIEVAL=onnx`) and a reranker/NLI scorer (`SEMANTIC_SCORER=onnx`). Both are **off unless enabled explicitly**, so a host bound to a fixed embedding model never uses a local one by accident.

## Provenance

Extracted from Unlimited Memory (2026). The extraction removed:
- the host application's server, database layer and administrative modules, where the memory code calls host services (persistence, admin audit, memory files), they are replaced by documented stand-ins;
- unrelated product modules and all non-memory features.

Tests: `cargo test --release --lib` (291 unit tests). Tests that needed the host application's HTTP test harness were not carried over.

License: Apache-2.0 (see the repository's `LICENSE` and `NOTICE`).
