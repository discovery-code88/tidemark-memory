//! Semantic retrieval channel: a rebuildable per-chunk embedding index.
//!
//! Lexical retrieval cannot find a record that answers a question in other
//! words. Each record is split into chunks (dialogue turns, paragraphs, or
//! packed sentences) and embedded once; a query scores a record by its best
//! matching chunk. The index is derived data keyed by record content hash and
//! embedding model: it never changes the database, is rebuilt when content or
//! model changes, and is persisted as a local cache so restarts do not
//! re-embed everything.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, RwLock};

use sha2::{Digest, Sha256};

use crate::semantic_scorer::{dot, OnnxEmbedder};

const CACHE_MAGIC: &[u8; 8] = b"UMSRI001";
const MAX_CHUNK_WORDS: usize = 80;

struct ChunkEntry {
    content_hash: u64,
    vectors: Arc<Vec<Vec<f32>>>,
}

pub(crate) struct SemanticRetrievalIndex {
    embedder: OnnxEmbedder,
    entries: RwLock<HashMap<u64, ChunkEntry>>,
    /// Passage vectors by text hash (for ranking evidence turns).
    passages: RwLock<HashMap<u64, Arc<Vec<f32>>>>,
    cache_path: Option<PathBuf>,
    sync_limit: usize,
    save_lock: Mutex<()>,
}

pub(crate) fn content_hash(content: &str) -> u64 {
    let digest = Sha256::digest(content.as_bytes());
    u64::from_le_bytes(digest[..8].try_into().expect("digest has 8 bytes"))
}

/// Projection header lines written by the source projector; they describe
/// provenance, not content.
fn is_projection_header(line: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "Universal source projection",
        "Archive type:",
        "Source file:",
        "Source:",
        "Evidence:",
        "Observation:",
        "Session date:",
    ];
    PREFIXES.iter().any(|prefix| line.starts_with(prefix))
}

/// Drops a leading turn locator such as `D1:3 ` so the chunk reads
/// `Caroline: ...`.
fn strip_turn_locator(line: &str) -> &str {
    let mut parts = line.splitn(2, ' ');
    let first = parts.next().unwrap_or_default();
    let rest = parts.next();
    let locator = first.len() <= 12
        && first.contains(':')
        && !first.ends_with(':')
        && first.chars().any(|character| character.is_ascii_digit())
        && first.chars().next().is_some_and(|character| character.is_ascii_alphabetic())
        && first
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == ':');
    match (locator, rest) {
        (true, Some(rest)) => rest,
        _ => line,
    }
}

/// Splits record content into embedding chunks: one per non-empty line or
/// paragraph, long ones packed by sentence up to `MAX_CHUNK_WORDS` words.
pub(crate) fn chunk_content(content: &str) -> Vec<String> {
    let mut chunks = Vec::new();
    for line in content.lines().map(str::trim) {
        if line.is_empty() || is_projection_header(line) {
            continue;
        }
        let line = strip_turn_locator(line);
        if line.split_whitespace().count() <= MAX_CHUNK_WORDS {
            chunks.push(line.to_string());
            continue;
        }
        let mut current = String::new();
        for sentence in line.split_inclusive(['.', '!', '?', '。', '！', '？']) {
            let words = current.split_whitespace().count() + sentence.split_whitespace().count();
            if words > MAX_CHUNK_WORDS && !current.is_empty() {
                chunks.push(std::mem::take(&mut current).trim().to_string());
            }
            current.push_str(sentence);
        }
        if !current.trim().is_empty() {
            chunks.push(current.trim().to_string());
        }
    }
    chunks
}

impl SemanticRetrievalIndex {
    fn new(embedder: OnnxEmbedder, cache_path: Option<PathBuf>, sync_limit: usize) -> Self {
        let index = Self {
            embedder,
            entries: RwLock::new(HashMap::new()),
            passages: RwLock::new(HashMap::new()),
            cache_path,
            sync_limit,
            save_lock: Mutex::new(()),
        };
        if let Some(path) = index.cache_path.clone() {
            match index.load_cache(&path) {
                Ok(count) if count > 0 => {
                    eprintln!("semantic retrieval cache loaded: {count} records from {}", path.display())
                }
                Ok(_) => {}
                Err(error) => eprintln!("semantic retrieval cache ignored: {error}"),
            }
        }
        index
    }

    pub(crate) fn model_id(&self) -> &str {
        self.embedder.model_id()
    }

    /// Embeds records that are missing or whose content changed.
    pub(crate) fn index_records(&self, records: &[(u64, &str)]) -> Result<usize, String> {
        let pending = {
            let entries = self.entries.read().map_err(|_| "semantic index poisoned")?;
            records
                .iter()
                .filter(|(id, content)| {
                    entries
                        .get(id)
                        .is_none_or(|entry| entry.content_hash != content_hash(content))
                })
                .map(|(id, content)| (*id, content_hash(content), chunk_content(content)))
                .collect::<Vec<_>>()
        };
        if pending.is_empty() {
            return Ok(0);
        }
        let texts = pending
            .iter()
            .flat_map(|(_, _, chunks)| chunks.iter().map(String::as_str))
            .collect::<Vec<_>>();
        let mut vectors = self.embedder.embed_passages(&texts)?.into_iter();
        let mut entries = self.entries.write().map_err(|_| "semantic index poisoned")?;
        for (id, hash, chunks) in &pending {
            let record_vectors = vectors.by_ref().take(chunks.len()).collect::<Vec<_>>();
            entries.insert(
                *id,
                ChunkEntry {
                    content_hash: *hash,
                    vectors: Arc::new(record_vectors),
                },
            );
        }
        drop(entries);
        if let Some(path) = &self.cache_path {
            if let Err(error) = self.save_cache(path) {
                eprintln!("semantic retrieval cache not saved: {error}");
            }
        }
        Ok(pending.len())
    }

    /// Best-chunk cosine similarity of each candidate record to the query,
    /// min-max normalized over the scored candidates (0 = least similar,
    /// 1 = most similar). Missing records are embedded first when there are
    /// at most `sync_limit` of them; otherwise the call returns `None` and
    /// retrieval stays lexical for this query.
    pub(crate) fn scores(&self, query: &str, records: &[(u64, &str)]) -> Option<HashMap<u64, f32>> {
        if records.is_empty() {
            return Some(HashMap::new());
        }
        let missing = {
            let entries = self.entries.read().ok()?;
            records
                .iter()
                .filter(|(id, content)| {
                    entries
                        .get(id)
                        .is_none_or(|entry| entry.content_hash != content_hash(content))
                })
                .count()
        };
        if missing > self.sync_limit {
            return None;
        }
        if missing > 0 {
            self.index_records(records).ok()?;
        }
        let query_vector = self.embedder.embed_queries(&[query]).ok()?.pop()?;
        let entries = self.entries.read().ok()?;
        let raw = records
            .iter()
            .filter_map(|(id, _)| {
                let entry = entries.get(id)?;
                let best = entry
                    .vectors
                    .iter()
                    .map(|vector| dot(&query_vector, vector))
                    .fold(f32::MIN, f32::max);
                (best > f32::MIN).then_some((*id, best))
            })
            .collect::<Vec<_>>();
        let low = raw.iter().map(|(_, score)| *score).fold(f32::MAX, f32::min);
        let high = raw.iter().map(|(_, score)| *score).fold(f32::MIN, f32::max);
        let span = (high - low).max(1e-6);
        Some(raw.into_iter().map(|(id, score)| (id, (score - low) / span)).collect())
    }

    /// Cosine similarity of the query to each passage; passage vectors are
    /// cached by text so a conversation's turns are embedded once.
    pub(crate) fn passage_similarities(&self, query: &str, texts: &[&str]) -> Option<Vec<f32>> {
        const PASSAGE_CACHE_LIMIT: usize = 500_000;
        let keys = texts.iter().map(|text| content_hash(text)).collect::<Vec<_>>();
        let missing = {
            let passages = self.passages.read().ok()?;
            let mut seen = std::collections::HashSet::new();
            keys.iter()
                .zip(texts)
                .filter(|(key, _)| !passages.contains_key(key) && seen.insert(**key))
                .map(|(key, text)| (*key, *text))
                .collect::<Vec<_>>()
        };
        if !missing.is_empty() {
            let vectors = self
                .embedder
                .embed_passages(&missing.iter().map(|(_, text)| *text).collect::<Vec<_>>())
                .ok()?;
            let mut passages = self.passages.write().ok()?;
            if passages.len() + missing.len() > PASSAGE_CACHE_LIMIT {
                passages.clear();
            }
            for ((key, _), vector) in missing.iter().zip(vectors) {
                passages.insert(*key, Arc::new(vector));
            }
        }
        let query_vector = self.embedder.embed_queries(&[query]).ok()?.pop()?;
        let passages = self.passages.read().ok()?;
        keys.iter()
            .map(|key| passages.get(key).map(|vector| dot(&query_vector, vector)))
            .collect()
    }

    fn save_cache(&self, path: &Path) -> Result<(), String> {
        let _guard = self.save_lock.lock().map_err(|_| "semantic cache lock poisoned")?;
        let entries = self.entries.read().map_err(|_| "semantic index poisoned")?;
        let mut buffer = Vec::new();
        buffer.extend_from_slice(CACHE_MAGIC);
        let model = self.model_id().as_bytes();
        buffer.extend_from_slice(&(model.len() as u32).to_le_bytes());
        buffer.extend_from_slice(model);
        buffer.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (id, entry) in entries.iter() {
            buffer.extend_from_slice(&id.to_le_bytes());
            buffer.extend_from_slice(&entry.content_hash.to_le_bytes());
            buffer.extend_from_slice(&(entry.vectors.len() as u32).to_le_bytes());
            let dims = entry.vectors.first().map(Vec::len).unwrap_or(0);
            buffer.extend_from_slice(&(dims as u32).to_le_bytes());
            for vector in entry.vectors.iter() {
                for value in vector {
                    buffer.extend_from_slice(&value.to_le_bytes());
                }
            }
        }
        drop(entries);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|error| error.to_string())?;
        }
        let temporary = path.with_extension("tmp");
        let mut file = std::fs::File::create(&temporary).map_err(|error| error.to_string())?;
        file.write_all(&buffer).map_err(|error| error.to_string())?;
        file.sync_all().map_err(|error| error.to_string())?;
        std::fs::rename(&temporary, path).map_err(|error| error.to_string())
    }

    fn load_cache(&self, path: &Path) -> Result<usize, String> {
        let mut bytes = Vec::new();
        match std::fs::File::open(path) {
            Ok(mut file) => file.read_to_end(&mut bytes).map_err(|error| error.to_string())?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(error) => return Err(error.to_string()),
        };
        let mut cursor = 0_usize;
        let mut take = |count: usize| -> Result<&[u8], String> {
            let slice = bytes
                .get(cursor..cursor + count)
                .ok_or("semantic cache truncated")?;
            cursor += count;
            Ok(slice)
        };
        if take(8)? != CACHE_MAGIC {
            return Err("semantic cache has an unknown format".into());
        }
        let model_len = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
        let model = String::from_utf8_lossy(take(model_len)?).to_string();
        if model != self.model_id() {
            return Err(format!("semantic cache built with {model}, not {}", self.model_id()));
        }
        let count = u64::from_le_bytes(take(8)?.try_into().unwrap()) as usize;
        let mut loaded = HashMap::with_capacity(count);
        for _ in 0..count {
            let id = u64::from_le_bytes(take(8)?.try_into().unwrap());
            let hash = u64::from_le_bytes(take(8)?.try_into().unwrap());
            let chunks = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let dims = u32::from_le_bytes(take(4)?.try_into().unwrap()) as usize;
            let mut vectors = Vec::with_capacity(chunks);
            for _ in 0..chunks {
                let raw = take(dims * 4)?;
                vectors.push(
                    raw.chunks_exact(4)
                        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                        .collect::<Vec<_>>(),
                );
            }
            loaded.insert(
                id,
                ChunkEntry {
                    content_hash: hash,
                    vectors: Arc::new(vectors),
                },
            );
        }
        let mut entries = self.entries.write().map_err(|_| "semantic index poisoned")?;
        let loaded_count = loaded.len();
        entries.extend(loaded);
        Ok(loaded_count)
    }
}

/// Weight of the normalized semantic score in the retrieval blend
/// (`SEMANTIC_RETRIEVAL_WEIGHT`).
pub(crate) fn retrieval_weight() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var("SEMANTIC_RETRIEVAL_WEIGHT")
            .ok()
            .and_then(|value| value.trim().parse::<f32>().ok())
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.15)
    })
}

fn build_from_env() -> Result<Option<SemanticRetrievalIndex>, String> {
    let mode = std::env::var("SEMANTIC_RETRIEVAL")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    let root = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/tidemark");
    let model_dir = std::env::var_os("SEMANTIC_EMBEDDING_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models/multilingual-e5-small"));
    match mode.as_str() {
        // Opt-in only: hosts with model rules (e.g. a fixed embedding model)
        // must not get a local model just because its files exist.
        "" | "off" | "none" | "disabled" => return Ok(None),
        "onnx" => {}
        other => return Err(format!("unsupported SEMANTIC_RETRIEVAL {other}")),
    }
    let threads = std::env::var("SEMANTIC_SCORER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(usize::from).unwrap_or(4))
        .max(1);
    let embedder = OnnxEmbedder::load(&model_dir, threads)?;
    let cache_path = match std::env::var("SEMANTIC_RETRIEVAL_CACHE") {
        Ok(value) if value.trim().is_empty() || value.trim() == "off" => None,
        Ok(value) => Some(PathBuf::from(value)),
        Err(_) => {
            let model = embedder.model_id().replace([':', '/'], "_");
            Some(root.join(format!("semantic-index/{model}.bin")))
        }
    };
    let sync_limit = std::env::var("SEMANTIC_RETRIEVAL_SYNC_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    Ok(Some(SemanticRetrievalIndex::new(embedder, cache_path, sync_limit)))
}

static INDEX: OnceLock<Option<SemanticRetrievalIndex>> = OnceLock::new();



/// Lets an ignored diagnostic test use the real index (normal tests never do).


/// Process-wide index; `None` when disabled, the model is missing, or it is
/// still loading in the background (retrieval then stays lexical). Tests never
/// load models unless a test explicitly opts in.
pub(crate) fn global_index() -> Option<&'static SemanticRetrievalIndex> {
    
    #[allow(unreachable_code)]
    INDEX.get().and_then(Option::as_ref)
}

/// Initializes the index (blocking); called by the startup loader.
pub(crate) fn global_index_blocking() -> Option<&'static SemanticRetrievalIndex> {
    INDEX
        .get_or_init(|| match build_from_env() {
            Ok(Some(index)) => {
                eprintln!("semantic retrieval loaded: {}", index.model_id());
                Some(index)
            }
            Ok(None) => None,
            Err(error) => {
                eprintln!("semantic retrieval unavailable: {error}");
                None
            }
        })
        .as_ref()
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_skip_projection_headers_and_turn_locators() {
        let content = "Universal source projection\nArchive type: source evidence\nSource file: x\n\nSession date: 1:56 pm on 8 May, 2023\n\nD1:3 Caroline: I went to a LGBTQ support group yesterday.\n\nD1:4 Melanie: Wow, that's cool!";
        assert_eq!(
            chunk_content(content),
            vec![
                "Caroline: I went to a LGBTQ support group yesterday.".to_string(),
                "Melanie: Wow, that's cool!".to_string()
            ]
        );
        let long = (0..30).map(|index| format!("Sentence number {index} is here.")).collect::<Vec<_>>().join(" ");
        let chunks = chunk_content(&long);
        assert!(chunks.len() > 1);
        assert!(chunks.iter().all(|chunk| chunk.split_whitespace().count() <= MAX_CHUNK_WORDS));
        assert_eq!(strip_turn_locator("Caroline: hi"), "Caroline: hi");
        assert_eq!(strip_turn_locator("D12:5 Caroline: hi"), "Caroline: hi");
    }

    #[test]
    fn content_hash_is_stable() {
        assert_eq!(content_hash("abc"), content_hash("abc"));
        assert_ne!(content_hash("abc"), content_hash("abd"));
    }
}
