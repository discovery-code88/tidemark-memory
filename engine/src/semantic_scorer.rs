//! Semantic relevance and entailment scoring.
//!
//! Lexical overlap cannot tell "Yesterday we started a road trip" apart from
//! an answer to "When did James start taking cooking classes?", and it rejects
//! paraphrases ("signed up for a cooking class"). A `SemanticScorer` answers
//! the two questions the answer pipeline actually needs:
//!
//! - relevance: does this passage answer this question? (cross-encoder reranker)
//! - entailment: does this source sentence support this claim? (NLI model)
//!
//! Backends are interchangeable. The default is an embedded ONNX Runtime
//! backend running small open models offline; deployments with their own
//! inference service can use the HTTP backend instead. Scores only decide
//! relevance and support; attribution, dates, negation and citation binding
//! stay deterministic in the claim gate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct EntailmentScores {
    pub(crate) entailment: f32,
    pub(crate) neutral: f32,
    pub(crate) contradiction: f32,
}

pub(crate) trait SemanticScorer: Send + Sync {
    /// Stable identifier of backend and models, recorded in receipts so a
    /// verdict can be traced to the model that produced it.
    fn backend_id(&self) -> &str;
    /// Probability in [0, 1] that each passage answers the question.
    fn relevance(&self, question: &str, passages: &[&str]) -> Result<Vec<f32>, String>;
    /// Entailment probabilities for (premise, hypothesis) pairs.
    fn entailment(&self, pairs: &[(&str, &str)]) -> Result<Vec<EntailmentScores>, String>;
    /// True when scoring sends evidence text off-host.
    fn remote(&self) -> bool {
        false
    }
}

const MAX_SEQUENCE_TOKENS: usize = 320;
const BATCH_SIZE: usize = 16;
const CACHE_LIMIT: usize = 2_000_000;

/// Opens an ONNX model. Full graph optimization of a base-size model takes
/// over a minute on a small CPU, so the optimized graph is saved next to the
/// model (`model.optimized.onnx`) on first load and reused afterwards. A
/// read-only model directory falls back to optimizing in memory.
pub(crate) fn load_session(model: &Path, threads: usize) -> Result<Session, String> {
    let optimized = model.with_file_name("model.optimized.onnx");
    let fresh = |path: &Path| {
        let modified = |path: &Path| std::fs::metadata(path).and_then(|meta| meta.modified()).ok();
        matches!((modified(path), modified(model)), (Some(cache), Some(source)) if cache >= source)
    };
    let builder = |level: GraphOptimizationLevel| -> Result<ort::session::builder::SessionBuilder, String> {
        Session::builder()
            .map_err(|error| error.to_string())?
            .with_optimization_level(level)
            .map_err(|error| error.to_string())?
            .with_intra_threads(threads)
            .map_err(|error| error.to_string())
    };
    if fresh(&optimized) {
        if let Ok(session) = builder(GraphOptimizationLevel::Disable)?.commit_from_file(&optimized) {
            return Ok(session);
        }
    }
    let saved = builder(GraphOptimizationLevel::Level3)?
        .with_optimized_model_path(&optimized)
        .map_err(|error| error.to_string())
        .and_then(|mut builder| {
            builder
                .commit_from_file(model)
                .map_err(|error| error.to_string())
        });
    match saved {
        Ok(session) => Ok(session),
        Err(_) => builder(GraphOptimizationLevel::Level3)?
            .commit_from_file(model)
            .map_err(|error| format!("{}: {error}", model.display())),
    }
}

/// A sequence-pair classifier (reranker or NLI head) run by ONNX Runtime.
struct CrossEncoder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
}

impl CrossEncoder {
    fn load(dir: &Path, threads: usize) -> Result<Self, String> {
        let model = dir.join("model.onnx");
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|error| format!("{}: {error}", dir.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: MAX_SEQUENCE_TOKENS,
                strategy: TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            }))
            .map_err(|error| error.to_string())?;
        tokenizer.with_padding(None);
        let session = load_session(&model, threads)?;
        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            input_names,
        })
    }

    /// Raw logits for each pair, in input order.
    fn logits(&self, pairs: &[(&str, &str)]) -> Result<Vec<Vec<f32>>, String> {
        let mut results = Vec::with_capacity(pairs.len());
        for chunk in pairs.chunks(BATCH_SIZE) {
            let encodings = self
                .tokenizer
                .encode_batch(
                    chunk
                        .iter()
                        .map(|(first, second)| (first.to_string(), second.to_string()))
                        .collect::<Vec<_>>(),
                    true,
                )
                .map_err(|error| error.to_string())?;
            let width = encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .max()
                .unwrap_or(1)
                .max(1);
            let rows = encodings.len();
            let mut ids = vec![0_i64; rows * width];
            let mut mask = vec![0_i64; rows * width];
            let mut types = vec![0_i64; rows * width];
            let pad_id = self
                .tokenizer
                .get_padding()
                .map(|padding| padding.pad_id as i64)
                .or_else(|| self.tokenizer.token_to_id("<pad>").map(i64::from))
                .or_else(|| self.tokenizer.token_to_id("[PAD]").map(i64::from))
                .unwrap_or(0);
            for (row, encoding) in encodings.iter().enumerate() {
                let offset = row * width;
                ids[offset..offset + width].fill(pad_id);
                for (column, ((id, attention), type_id)) in encoding
                    .get_ids()
                    .iter()
                    .zip(encoding.get_attention_mask())
                    .zip(encoding.get_type_ids())
                    .enumerate()
                {
                    ids[offset + column] = i64::from(*id);
                    mask[offset + column] = i64::from(*attention);
                    types[offset + column] = i64::from(*type_id);
                }
            }
            let shape = [rows as i64, width as i64];
            let mut inputs = Vec::new();
            for name in &self.input_names {
                let data = match name.as_str() {
                    "input_ids" => ids.clone(),
                    "attention_mask" => mask.clone(),
                    "token_type_ids" => types.clone(),
                    other => return Err(format!("unsupported model input {other}")),
                };
                let tensor = Tensor::from_array((shape, data)).map_err(|error| error.to_string())?;
                inputs.push((name.clone(), tensor.into_dyn()));
            }
            let mut session = self
                .session
                .lock()
                .map_err(|_| "semantic scorer session poisoned".to_string())?;
            let outputs = session.run(inputs).map_err(|error| error.to_string())?;
            let (out_shape, values) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|error| error.to_string())?;
            let labels = out_shape.last().copied().unwrap_or(1).max(1) as usize;
            if values.len() != rows * labels {
                return Err(format!(
                    "unexpected output size {} for {rows} rows x {labels} labels",
                    values.len()
                ));
            }
            results.extend(values.chunks(labels).map(<[f32]>::to_vec));
        }
        Ok(results)
    }
}

/// Sentence embedding model (mean pooling, L2-normalized) run by ONNX
/// Runtime. E5-family models expect `query: ` / `passage: ` prefixes.
pub(crate) struct OnnxEmbedder {
    id: String,
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    query_prefix: String,
    passage_prefix: String,
}

impl OnnxEmbedder {
    pub(crate) fn load(dir: &Path, threads: usize) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|error| format!("{}: {error}", dir.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: 256,
                strategy: TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            }))
            .map_err(|error| error.to_string())?;
        tokenizer.with_padding(None);
        let model = dir.join("model.onnx");
        let session = load_session(&model, threads)?;
        let input_names = session
            .inputs()
            .iter()
            .map(|input| input.name().to_string())
            .collect();
        let name = dir
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_default();
        let e5 = name.contains("e5");
        Ok(Self {
            id: format!("onnx:{name}"),
            session: Mutex::new(session),
            tokenizer,
            input_names,
            query_prefix: if e5 { "query: ".into() } else { String::new() },
            passage_prefix: if e5 { "passage: ".into() } else { String::new() },
        })
    }

    pub(crate) fn model_id(&self) -> &str {
        &self.id
    }

    pub(crate) fn embed_queries(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.embed(&self.query_prefix, texts)
    }

    pub(crate) fn embed_passages(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        self.embed(&self.passage_prefix, texts)
    }

    fn embed(&self, prefix: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>, String> {
        let mut vectors = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(32) {
            let encodings = self
                .tokenizer
                .encode_batch(
                    chunk.iter().map(|text| format!("{prefix}{text}")).collect::<Vec<_>>(),
                    true,
                )
                .map_err(|error| error.to_string())?;
            let width = encodings
                .iter()
                .map(|encoding| encoding.get_ids().len())
                .max()
                .unwrap_or(1)
                .max(1);
            let rows = encodings.len();
            let mut ids = vec![0_i64; rows * width];
            let mut mask = vec![0_i64; rows * width];
            let mut types = vec![0_i64; rows * width];
            let pad_id = self
                .tokenizer
                .token_to_id("<pad>")
                .or_else(|| self.tokenizer.token_to_id("[PAD]"))
                .map(i64::from)
                .unwrap_or(0);
            for (row, encoding) in encodings.iter().enumerate() {
                let offset = row * width;
                ids[offset..offset + width].fill(pad_id);
                for (column, ((id, attention), type_id)) in encoding
                    .get_ids()
                    .iter()
                    .zip(encoding.get_attention_mask())
                    .zip(encoding.get_type_ids())
                    .enumerate()
                {
                    ids[offset + column] = i64::from(*id);
                    mask[offset + column] = i64::from(*attention);
                    types[offset + column] = i64::from(*type_id);
                }
            }
            let shape = [rows as i64, width as i64];
            let mut inputs = Vec::new();
            for name in &self.input_names {
                let data = match name.as_str() {
                    "input_ids" => ids.clone(),
                    "attention_mask" => mask.clone(),
                    "token_type_ids" => types.clone(),
                    other => return Err(format!("unsupported model input {other}")),
                };
                let tensor = Tensor::from_array((shape, data)).map_err(|error| error.to_string())?;
                inputs.push((name.clone(), tensor.into_dyn()));
            }
            let mut session = self
                .session
                .lock()
                .map_err(|_| "embedder session poisoned".to_string())?;
            let outputs = session.run(inputs).map_err(|error| error.to_string())?;
            let (out_shape, values) = outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(|error| error.to_string())?;
            let dims = out_shape.last().copied().unwrap_or(0).max(0) as usize;
            if dims == 0 || values.len() != rows * width * dims {
                return Err(format!("unexpected embedding output shape {out_shape:?}"));
            }
            for row in 0..rows {
                let mut pooled = vec![0.0_f32; dims];
                let mut count = 0.0_f32;
                for column in 0..width {
                    if mask[row * width + column] == 0 {
                        continue;
                    }
                    count += 1.0;
                    let start = (row * width + column) * dims;
                    for (sum, value) in pooled.iter_mut().zip(&values[start..start + dims]) {
                        *sum += value;
                    }
                }
                let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt().max(1e-12);
                let scale = if count > 0.0 { 1.0 / norm } else { 0.0 };
                vectors.push(pooled.into_iter().map(|value| value * scale).collect());
            }
        }
        Ok(vectors)
    }
}

pub(crate) fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

/// Label order of the NLI head, read from the model's config.json.
#[derive(Debug, Clone, Copy)]
struct NliLabels {
    entailment: usize,
    neutral: usize,
    contradiction: usize,
}

impl NliLabels {
    fn from_config(dir: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|error| format!("{}: {error}", dir.display()))?;
        let config: serde_json::Value =
            serde_json::from_str(&text).map_err(|error| error.to_string())?;
        let labels = config
            .get("id2label")
            .and_then(serde_json::Value::as_object)
            .ok_or("NLI config.json has no id2label")?;
        let find = |wanted: &str| {
            labels
                .iter()
                .find(|(_, label)| {
                    label
                        .as_str()
                        .is_some_and(|label| label.to_lowercase().starts_with(wanted))
                })
                .and_then(|(index, _)| index.parse::<usize>().ok())
                .ok_or_else(|| format!("NLI config.json has no {wanted} label"))
        };
        Ok(Self {
            entailment: find("entail")?,
            neutral: find("neutral")?,
            contradiction: find("contradict")?,
        })
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps = values.iter().map(|value| (value - max).exp()).collect::<Vec<_>>();
    let total = exps.iter().sum::<f32>();
    exps.into_iter().map(|value| value / total).collect()
}

/// Stable across processes and builds, so scores can be persisted. Only the
/// digest is stored, never the text.
fn pair_key(kind: u8, first: &str, second: &str) -> u64 {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update([kind]);
    hasher.update((first.len() as u64).to_le_bytes());
    hasher.update(first.as_bytes());
    hasher.update(second.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("digest has 8 bytes"))
}

/// Memoizes pair scores: the pipeline re-checks the same pairs across
/// planning, selection and claim verification. With a cache file, scores
/// survive restarts: re-answering a question over unchanged records reuses
/// every model call. The file holds (digest, scores) records only.
struct ScoreCache {
    entries: Mutex<HashMap<u64, Vec<f32>>>,
    log: Option<Mutex<std::io::BufWriter<std::fs::File>>>,
}

impl ScoreCache {
    fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            log: None,
        }
    }

    /// Loads `path` (if present) and appends newly computed scores to it.
    fn persistent(path: &Path) -> Self {
        use std::io::Read;
        let mut entries = HashMap::new();
        if let Ok(mut file) = std::fs::File::open(path) {
            let mut bytes = Vec::new();
            if file.read_to_end(&mut bytes).is_ok() {
                let mut cursor = 0;
                while cursor + 9 <= bytes.len() {
                    let key = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
                    let count = bytes[cursor + 8] as usize;
                    let end = cursor + 9 + count * 4;
                    if end > bytes.len() {
                        break; // torn final record from an interrupted write
                    }
                    let values = bytes[cursor + 9..end]
                        .chunks_exact(4)
                        .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                        .collect::<Vec<_>>();
                    entries.insert(key, values);
                    cursor = end;
                }
            }
        }
        let log = path
            .parent()
            .map(std::fs::create_dir_all)
            .transpose()
            .ok()
            .and_then(|_| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                    .ok()
            })
            .map(|file| Mutex::new(std::io::BufWriter::new(file)));
        if !entries.is_empty() {
            eprintln!("semantic score cache loaded: {} pairs from {}", entries.len(), path.display());
        }
        Self {
            entries: Mutex::new(entries),
            log,
        }
    }

    fn scores(
        &self,
        kind: u8,
        pairs: &[(&str, &str)],
        compute: impl FnOnce(&[(&str, &str)]) -> Result<Vec<Vec<f32>>, String>,
    ) -> Result<Vec<Vec<f32>>, String> {
        let keys = pairs
            .iter()
            .map(|(first, second)| pair_key(kind, first, second))
            .collect::<Vec<_>>();
        let mut results = vec![None; pairs.len()];
        let mut missing = Vec::new();
        {
            let entries = self.entries.lock().map_err(|_| "score cache poisoned")?;
            for (index, key) in keys.iter().enumerate() {
                match entries.get(key) {
                    Some(value) => results[index] = Some(value.clone()),
                    None => missing.push(index),
                }
            }
        }
        if !missing.is_empty() {
            let wanted = missing.iter().map(|index| pairs[*index]).collect::<Vec<_>>();
            let computed = compute(&wanted)?;
            let mut entries = self.entries.lock().map_err(|_| "score cache poisoned")?;
            if entries.len() + computed.len() > CACHE_LIMIT {
                entries.clear();
            }
            let mut record = Vec::new();
            for (index, value) in missing.into_iter().zip(computed) {
                if self.log.is_some() && value.len() <= u8::MAX as usize {
                    record.extend_from_slice(&keys[index].to_le_bytes());
                    record.push(value.len() as u8);
                    for score in &value {
                        record.extend_from_slice(&score.to_le_bytes());
                    }
                }
                entries.insert(keys[index], value.clone());
                results[index] = Some(value);
            }
            if let Some(log) = &self.log {
                use std::io::Write;
                if let Ok(mut log) = log.lock() {
                    let _ = log.write_all(&record).and_then(|_| log.flush());
                }
            }
        }
        Ok(results.into_iter().map(Option::unwrap_or_default).collect())
    }
}

/// Score cache file for a backend (`SEMANTIC_SCORE_CACHE`: a directory, or
/// `off`). Default: `~/.cache/tidemark/score-cache/`.
fn score_cache_for(backend_id: &str) -> ScoreCache {
    let directory = match std::env::var("SEMANTIC_SCORE_CACHE") {
        Ok(value) if value.trim() == "off" => return ScoreCache::new(),
        Ok(value) if !value.trim().is_empty() => PathBuf::from(value.trim()),
        _ => match std::env::var_os("HOME") {
            Some(home) => PathBuf::from(home).join(".cache/tidemark/score-cache"),
            None => return ScoreCache::new(),
        },
    };
    let name = backend_id.replace(|ch: char| !ch.is_ascii_alphanumeric() && ch != '-', "_");
    ScoreCache::persistent(&directory.join(format!("{name}.bin")))
}

/// Opt-in per-call cost trace (`SEMANTIC_SCORER_TRACE=1`).
fn trace_call(kind: &str, pairs: usize, computed: usize, started: std::time::Instant) {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    if *ENABLED.get_or_init(|| std::env::var("SEMANTIC_SCORER_TRACE").is_ok_and(|value| value == "1")) {
        eprintln!(
            "{{\"event\":\"semantic_scorer_call\",\"kind\":\"{kind}\",\"pairs\":{pairs},\"computed\":{computed},\"ms\":{}}}",
            started.elapsed().as_millis()
        );
    }
}

pub(crate) struct OnnxSemanticScorer {
    id: String,
    reranker: CrossEncoder,
    nli: CrossEncoder,
    nli_labels: NliLabels,
    cache: ScoreCache,
}

impl OnnxSemanticScorer {
    pub(crate) fn load(reranker_dir: &Path, nli_dir: &Path, threads: usize) -> Result<Self, String> {
        let name = |dir: &Path| {
            dir.file_name()
                .map(|name| name.to_string_lossy().to_string())
                .unwrap_or_default()
        };
        let id = format!("onnx:{}+{}", name(reranker_dir), name(nli_dir));
        Ok(Self {
            reranker: CrossEncoder::load(reranker_dir, threads)?,
            nli: CrossEncoder::load(nli_dir, threads)?,
            nli_labels: NliLabels::from_config(nli_dir)?,
            cache: score_cache_for(&id),
            id,
        })
    }
}

impl SemanticScorer for OnnxSemanticScorer {
    fn backend_id(&self) -> &str {
        &self.id
    }

    fn relevance(&self, question: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        let pairs = passages
            .iter()
            .map(|passage| (question, *passage))
            .collect::<Vec<_>>();
        let started = std::time::Instant::now();
        let mut computed = 0;
        let logits = self.cache.scores(0, &pairs, |pairs| {
            computed = pairs.len();
            self.reranker.logits(pairs)
        })?;
        trace_call("relevance", pairs.len(), computed, started);
        Ok(logits
            .iter()
            .map(|row| row.first().copied().map(sigmoid).unwrap_or(0.0))
            .collect())
    }

    fn entailment(&self, pairs: &[(&str, &str)]) -> Result<Vec<EntailmentScores>, String> {
        let started = std::time::Instant::now();
        let mut computed = 0;
        let logits = self.cache.scores(1, pairs, |pairs| {
            computed = pairs.len();
            self.nli.logits(pairs)
        })?;
        trace_call("entailment", pairs.len(), computed, started);
        let labels = self.nli_labels;
        logits
            .iter()
            .map(|row| {
                let probabilities = softmax(row);
                let get = |index: usize| {
                    probabilities
                        .get(index)
                        .copied()
                        .ok_or_else(|| format!("NLI output has no label {index}"))
                };
                Ok(EntailmentScores {
                    entailment: get(labels.entailment)?,
                    neutral: get(labels.neutral)?,
                    contradiction: get(labels.contradiction)?,
                })
            })
            .collect()
    }
}

fn default_model_root() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache/tidemark/models")
}

fn env_path(name: &str, default: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| default_model_root().join(default))
}

/// Builds the configured scorer. `SEMANTIC_SCORER`: `onnx` (default when the
/// model files exist), `off`.
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn scorer_from_env() -> Result<Option<Arc<dyn SemanticScorer>>, String> {
    let backend = std::env::var("SEMANTIC_SCORER")
        .unwrap_or_default()
        .trim()
        .to_lowercase();
    let reranker = env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-v2-m3");
    let nli = env_path("SEMANTIC_NLI_MODEL_DIR", "mdeberta-v3-base-xnli");
    let threads = std::env::var("SEMANTIC_SCORER_THREADS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(4)
        })
        .max(1);
    match backend.as_str() {
        // Opt-in only (see semantic_retrieval::build_from_env).
        "" | "off" | "none" | "disabled" => Ok(None),
        "onnx" => Ok(Some(with_escalation(Arc::new(OnnxSemanticScorer::load(
            &reranker, &nli, threads,
        )?))?)),
        "http" => Ok(Some(with_escalation(Arc::new(HttpSemanticScorer::from_env()?))?)),
        other => Err(format!("unsupported SEMANTIC_SCORER backend {other}")),
    }
}

/// Wraps the scorer with the LLM judge when `SEMANTIC_ESCALATION=llm`.
fn with_escalation(scorer: Arc<dyn SemanticScorer>) -> Result<Arc<dyn SemanticScorer>, String> {
    match std::env::var("SEMANTIC_ESCALATION")
        .unwrap_or_default()
        .trim()
        .to_lowercase()
        .as_str()
    {
        "" | "off" | "none" => Ok(scorer),
        "llm" => Ok(Arc::new(JudgedScorer::from_env(scorer)?)),
        other => Err(format!("unsupported SEMANTIC_ESCALATION {other}")),
    }
}

// ---------------------------------------------------------------------------
// Request scope for remote backends

thread_local! {
    static REQUEST_SCOPE: std::cell::RefCell<Option<crate::embedding::EmbeddingScope>> =
        const { std::cell::RefCell::new(None) };
}

/// Runs `work` with the tenant/workspace scope and provider approval of the
/// answer request. Remote backends send evidence text through the provider
/// egress gateway, which enforces the workspace policy for that scope; without
/// a scope they fail closed and the pipeline keeps its lexical checks.
pub(crate) fn with_request_scope<T>(
    scope: Option<crate::embedding::EmbeddingScope>,
    work: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<crate::embedding::EmbeddingScope>);
    impl Drop for Restore {
        fn drop(&mut self) {
            REQUEST_SCOPE.with(|slot| *slot.borrow_mut() = self.0.take());
        }
    }
    let previous = REQUEST_SCOPE.with(|slot| slot.replace(scope));
    let _restore = Restore(previous);
    work()
}

fn request_scope() -> Option<crate::embedding::EmbeddingScope> {
    REQUEST_SCOPE.with(|slot| slot.borrow().clone())
}

/// True when the configured scorer sends data off-host, so handlers must
/// resolve the request scope before answering.
pub(crate) fn scorer_requires_request_scope() -> bool {
    #[cfg(test)]
    {
        false
    }
    #[cfg(not(test))]
    {
        SCORER.get().cloned().flatten().is_some_and(|scorer| scorer.remote())
    }
}

fn remaining_budget(default_ms: u64) -> std::time::Duration {
    let limit = std::time::Duration::from_millis(default_ms);
    crate::answer_execution::answer_worker_deadline()
        .map(|deadline| deadline.saturating_duration_since(tokio::time::Instant::now()).min(limit))
        .unwrap_or(limit)
}

/// POSTs JSON through the provider egress gateway from synchronous pipeline
/// code. The request runs on a scoped thread with its own runtime, so this is
/// safe whether or not the caller is inside a Tokio context.
fn egress_post_json(
    gateway: &crate::provider_egress::ProviderEgressGateway,
    url: &str,
    headers: &[(String, String)],
    purpose: &str,
    provider: &str,
    model: &str,
    payload: serde_json::Value,
    timeout_ms: u64,
) -> Result<serde_json::Value, String> {
    let scope = request_scope().ok_or("semantic_scorer_request_scope_missing")?;
    let budget = remaining_budget(timeout_ms);
    if budget.is_zero() {
        return Err("semantic_scorer_deadline_exceeded".into());
    }
    let request = crate::provider_egress::ProviderEgressRequest::from_env(
        "service:semantic-scorer",
        purpose,
        provider,
        model,
        "text",
        &["text", "memory_evidence"],
        payload,
    )
    .with_workspace_scope(&scope.tenant_id, &scope.workspace_id)
    .with_explicit_user_approval(scope.explicit_provider_approval);
    std::thread::scope(|threads| {
        threads
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .map_err(|error| error.to_string())?;
                runtime.block_on(async {
                    let work = async {
                        let (response, _) = gateway.post_json_with_headers(url, headers, request).await?;
                        let response = gateway.read_json_bounded(response, None).await?;
                        if !(200..300).contains(&response.status) {
                            return Err(format!("semantic_scorer_http_status:{}", response.status));
                        }
                        Ok(response.payload)
                    };
                    tokio::time::timeout(budget, work)
                        .await
                        .map_err(|_| "semantic_scorer_deadline_exceeded".to_string())?
                })
            })
            .join()
            .map_err(|_| "semantic_scorer_request_panicked".to_string())?
    })
}

/// Bearer auth for rerank/NLI inference servers.
fn bearer_headers(key: Option<&str>) -> Vec<(String, String)> {
    crate::llm_protocol::LlmProtocol::OpenAi.auth_headers(key)
}

// ---------------------------------------------------------------------------
// HTTP backend: business inference servers and vendor rerank APIs

#[derive(Debug, Clone, Copy, PartialEq)]
enum RerankApi {
    /// Hugging Face Text Embeddings Inference `/rerank`.
    Tei,
    /// Cohere-style `/rerank` (Cohere, Jina, Voyage, Mixedbread, vLLM).
    Cohere,
}

pub(crate) struct HttpSemanticScorer {
    id: String,
    gateway: crate::provider_egress::ProviderEgressGateway,
    rerank_url: String,
    rerank_api: RerankApi,
    rerank_model: String,
    nli_url: Option<String>,
    api_key: Option<String>,
    timeout_ms: u64,
    cache: ScoreCache,
}

fn env_string(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

impl HttpSemanticScorer {
    fn from_env() -> Result<Self, String> {
        let rerank_url =
            env_string("SEMANTIC_RERANK_URL").ok_or("SEMANTIC_RERANK_URL is required for http")?;
        let rerank_model = env_string("SEMANTIC_RERANK_MODEL").unwrap_or_default();
        let rerank_api = match env_string("SEMANTIC_RERANK_API").as_deref() {
            Some("tei") => RerankApi::Tei,
            Some("cohere") => RerankApi::Cohere,
            None if rerank_model.is_empty() => RerankApi::Tei,
            None => RerankApi::Cohere,
            Some(other) => return Err(format!("unsupported SEMANTIC_RERANK_API {other}")),
        };
        Ok(Self {
            id: format!(
                "http:{}:{}",
                if rerank_api == RerankApi::Tei { "tei" } else { "cohere" },
                if rerank_model.is_empty() { "server-default" } else { &rerank_model }
            ),
            gateway: crate::provider_egress::ProviderEgressGateway::from_env(),
            rerank_url,
            rerank_api,
            rerank_model,
            nli_url: env_string("SEMANTIC_NLI_URL"),
            api_key: env_string("SEMANTIC_SCORER_API_KEY"),
            timeout_ms: std::env::var("SEMANTIC_SCORER_TIMEOUT_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(8_000),
            cache: ScoreCache::new(),
        })
    }
}

fn rerank_payload(api: RerankApi, model: &str, question: &str, passages: &[&str]) -> serde_json::Value {
    match api {
        RerankApi::Tei => serde_json::json!({
            "query": question, "texts": passages, "raw_scores": false, "truncate": true
        }),
        RerankApi::Cohere => serde_json::json!({
            "model": model, "query": question, "documents": passages,
            "top_n": passages.len(), "return_documents": false
        }),
    }
}

/// Scores in passage order from a TEI (`[{index, score}]`) or Cohere-style
/// (`{results|data: [{index, relevance_score}]}`) response.
fn parse_rerank_response(value: &serde_json::Value, count: usize) -> Result<Vec<f32>, String> {
    let rows = value
        .as_array()
        .or_else(|| value.get("results").and_then(serde_json::Value::as_array))
        .or_else(|| value.get("data").and_then(serde_json::Value::as_array))
        .ok_or("rerank response has no result list")?;
    let mut scores = vec![None; count];
    for row in rows {
        let index = row
            .get("index")
            .and_then(serde_json::Value::as_u64)
            .ok_or("rerank result has no index")? as usize;
        let score = row
            .get("relevance_score")
            .or_else(|| row.get("score"))
            .and_then(serde_json::Value::as_f64)
            .ok_or("rerank result has no score")? as f32;
        *scores.get_mut(index).ok_or("rerank result index out of range")? = Some(score);
    }
    scores
        .into_iter()
        .map(|score| score.ok_or_else(|| "rerank response omitted a passage".to_string()))
        .collect()
}

/// TEI `/predict` for pair inputs: one list of `{label, score}` per pair.
fn parse_nli_response(value: &serde_json::Value, count: usize) -> Result<Vec<EntailmentScores>, String> {
    let rows = value.as_array().ok_or("NLI response is not a list")?;
    if rows.len() != count {
        return Err(format!("NLI response has {} rows for {count} pairs", rows.len()));
    }
    rows.iter()
        .map(|row| {
            let labels = row.as_array().ok_or("NLI row is not a label list")?;
            let find = |wanted: &str| {
                labels
                    .iter()
                    .find(|label| {
                        label
                            .get("label")
                            .and_then(serde_json::Value::as_str)
                            .is_some_and(|label| label.to_lowercase().starts_with(wanted))
                    })
                    .and_then(|label| label.get("score").and_then(serde_json::Value::as_f64))
                    .map(|score| score as f32)
                    .ok_or_else(|| format!("NLI row has no {wanted} label"))
            };
            Ok(EntailmentScores {
                entailment: find("entail")?,
                neutral: find("neutral")?,
                contradiction: find("contradict")?,
            })
        })
        .collect()
}

impl SemanticScorer for HttpSemanticScorer {
    fn backend_id(&self) -> &str {
        &self.id
    }

    fn remote(&self) -> bool {
        true
    }

    fn relevance(&self, question: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        let pairs = passages
            .iter()
            .map(|passage| (question, *passage))
            .collect::<Vec<_>>();
        let scores = self.cache.scores(0, &pairs, |pairs| {
            let passages = pairs.iter().map(|(_, passage)| *passage).collect::<Vec<_>>();
            let response = egress_post_json(
                &self.gateway,
                &self.rerank_url,
                &bearer_headers(self.api_key.as_deref()),
                "answer_semantic_relevance",
                "semantic_rerank_http",
                &self.rerank_model,
                rerank_payload(self.rerank_api, &self.rerank_model, question, &passages),
                self.timeout_ms,
            )?;
            Ok(parse_rerank_response(&response, passages.len())?
                .into_iter()
                .map(|score| vec![score])
                .collect())
        })?;
        Ok(scores.iter().map(|row| row.first().copied().unwrap_or(0.0)).collect())
    }

    fn entailment(&self, pairs: &[(&str, &str)]) -> Result<Vec<EntailmentScores>, String> {
        let url = self.nli_url.as_deref().ok_or("SEMANTIC_NLI_URL not configured")?;
        let rows = self.cache.scores(1, pairs, |pairs| {
            let response = egress_post_json(
                &self.gateway,
                url,
                &bearer_headers(self.api_key.as_deref()),
                "answer_semantic_entailment",
                "semantic_nli_http",
                "server-default",
                serde_json::json!({
                    "inputs": pairs.iter().map(|(premise, hypothesis)| [premise, hypothesis]).collect::<Vec<_>>(),
                    "raw_scores": false,
                    "truncate": true
                }),
                self.timeout_ms,
            )?;
            Ok(parse_nli_response(&response, pairs.len())?
                .into_iter()
                .map(|scores| vec![scores.entailment, scores.neutral, scores.contradiction])
                .collect())
        })?;
        Ok(rows
            .iter()
            .map(|row| EntailmentScores {
                entailment: row.first().copied().unwrap_or(0.0),
                neutral: row.get(1).copied().unwrap_or(0.0),
                contradiction: row.get(2).copied().unwrap_or(0.0),
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// LLM judge escalation for uncertain relevance

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct JudgeVerdicts {
    verdicts: Vec<JudgeVerdict>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct JudgeVerdict {
    index: u32,
    verdict: JudgeLabel,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum JudgeLabel {
    Answers,
    DoesNotAnswer,
    Unsure,
}

const JUDGE_PROMPT: &str = "You judge whether passages from a person's conversation history answer a question. \
A passage answers only if it states the fact the question asks for, possibly in different words, about the person the question names. \
A passage that is merely on the same topic, about someone else, or only a reaction does not answer. \
Return one verdict per passage index: answers, does_not_answer, or unsure.";

/// Scores in the uncertain middle band are escalated to an LLM judge in one
/// batched, schema-constrained call. Confident scores are never sent.
pub(crate) struct JudgedScorer {
    inner: Arc<dyn SemanticScorer>,
    id: String,
    gateway: crate::provider_egress::ProviderEgressGateway,
    endpoint: crate::llm_protocol::LlmEndpoint,
    url: String,
    model: String,
    api_key: Option<String>,
    max_passages: usize,
    timeout_ms: u64,
    cache: ScoreCache,
}

impl JudgedScorer {
    fn from_env(inner: Arc<dyn SemanticScorer>) -> Result<Self, String> {
        let base = env_string("SEMANTIC_JUDGE_BASE_URL").ok_or("SEMANTIC_JUDGE_BASE_URL is required")?;
        let model = env_string("SEMANTIC_JUDGE_MODEL").ok_or("SEMANTIC_JUDGE_MODEL is required")?;
        let base = base.trim_end_matches('/');
        let probe = crate::llm_protocol::LlmEndpoint::from_env("SEMANTIC_JUDGE", base)?;
        // The judge has always taken an OpenAI base URL; other protocols take
        // their own base or full endpoint.
        let base = match probe.protocol_name() {
            "openai_compat" if !base.ends_with("/chat/completions") => format!("{base}/chat/completions"),
            _ => base.to_string(),
        };
        let endpoint = crate::llm_protocol::LlmEndpoint::from_env("SEMANTIC_JUDGE", &base)?;
        let url = endpoint.endpoint(&model);
        Ok(Self {
            id: format!("{}+judge:{model}", inner.backend_id()),
            inner,
            gateway: crate::provider_egress::ProviderEgressGateway::from_env(),
            endpoint,
            url,
            model,
            api_key: env_string("SEMANTIC_JUDGE_API_KEY"),
            max_passages: std::env::var("SEMANTIC_JUDGE_MAX_PASSAGES")
                .ok()
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(12),
            timeout_ms: std::env::var("SEMANTIC_JUDGE_TIMEOUT_MS")
                .ok()
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(15_000),
            cache: ScoreCache::new(),
        })
    }

    fn judge(&self, question: &str, passages: &[&str]) -> Result<Vec<JudgeLabel>, String> {
        let schema = crate::answer_reasoning_provider::strict_response_format::<JudgeVerdicts>("relevance_verdicts")
            ["json_schema"]["schema"].clone();
        let payload = self.endpoint.body(&crate::llm_protocol::LlmRequest {
            model: &self.model,
            system: JUDGE_PROMPT,
            user: serde_json::json!({
                "question": question,
                "passages": passages.iter().enumerate()
                    .map(|(index, text)| serde_json::json!({"index": index, "text": text}))
                    .collect::<Vec<_>>(),
            }).to_string(),
            temperature: 0.0,
            json: true,
            schema: Some(("relevance_verdicts", schema)),
            strict: true,
            native_json: self.endpoint.supports_native_json() && crate::llm_protocol::native_json_allowed(&self.url),
            max_tokens: 4096,
        });
        let response = egress_post_json(
            &self.gateway,
            &self.url,
            &self.endpoint.headers(self.api_key.as_deref()),
            "answer_semantic_relevance_judge",
            self.endpoint.protocol_name(),
            &self.model,
            payload,
            self.timeout_ms,
        )?;
        parse_judge_response(&self.endpoint, &response, passages.len())
    }
}

fn parse_judge_response(
    endpoint: &crate::llm_protocol::LlmEndpoint,
    value: &serde_json::Value,
    count: usize,
) -> Result<Vec<JudgeLabel>, String> {
    let content = endpoint
        .extract_text(value)
        .map_err(|error| format!("judge response has no content: {error}"))?;
    let verdicts: JudgeVerdicts = crate::llm_protocol::extract_json(&content)
        .and_then(|value| serde_json::from_value(value).map_err(|error| error.to_string())).map_err(|error| format!("judge contract invalid: {error}"))?;
    let mut labels = vec![JudgeLabel::Unsure; count];
    for verdict in verdicts.verdicts {
        if let Some(label) = labels.get_mut(verdict.index as usize) {
            *label = verdict.verdict;
        }
    }
    Ok(labels)
}

impl SemanticScorer for JudgedScorer {
    fn backend_id(&self) -> &str {
        &self.id
    }

    fn remote(&self) -> bool {
        true
    }

    fn relevance(&self, question: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        let mut scores = self.inner.relevance(question, passages)?;
        let (grant, revoke) = (grant_threshold(), revoke_threshold());
        let mut uncertain = scores
            .iter()
            .enumerate()
            .filter(|(_, score)| **score >= revoke && **score < grant)
            .map(|(index, score)| (index, *score))
            .collect::<Vec<_>>();
        uncertain.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap_or(std::cmp::Ordering::Equal));
        uncertain.truncate(self.max_passages);
        if uncertain.is_empty() {
            return Ok(scores);
        }
        let pairs = uncertain
            .iter()
            .map(|(index, _)| (question, passages[*index]))
            .collect::<Vec<_>>();
        // Verdicts are cached as 1.0 / 0.0 / 0.5; a failed judge call leaves
        // the uncertain scores unchanged, which defers to the lexical checks.
        let verdicts = self.cache.scores(2, &pairs, |pairs| {
            let texts = pairs.iter().map(|(_, passage)| *passage).collect::<Vec<_>>();
            Ok(self
                .judge(question, &texts)?
                .into_iter()
                .map(|label| {
                    vec![match label {
                        JudgeLabel::Answers => 1.0,
                        JudgeLabel::DoesNotAnswer => 0.0,
                        JudgeLabel::Unsure => 0.5,
                    }]
                })
                .collect())
        });
        if let Ok(verdicts) = verdicts {
            for ((index, score), verdict) in uncertain.into_iter().zip(verdicts) {
                match verdict.first().copied() {
                    Some(value) if value >= 0.99 => scores[index] = score.max(grant),
                    Some(value) if value <= 0.01 => scores[index] = score.min(revoke * 0.5),
                    _ => {}
                }
            }
        }
        Ok(scores)
    }

    fn entailment(&self, pairs: &[(&str, &str)]) -> Result<Vec<EntailmentScores>, String> {
        self.inner.entailment(pairs)
    }
}

/// Relevance at or above this grants answer/action/object slots that lexical
/// matching missed. Calibrated on LoCoMo dev conversations only
/// (bge-reranker-v2-m3, turn plus preceding turn): ~58% of gold turns reach
/// it against ~4% of the most word-similar other turns.
pub(crate) const RELEVANCE_GRANT: f32 = 0.5;
/// Relevance below this revokes lexically granted slots: ~93% of gold turns
/// stay above it while ~74% of word-similar distractors fall below.
pub(crate) const RELEVANCE_REVOKE: f32 = 0.01;
/// A claim whose source contradicts it with this NLI probability is vetoed.
pub(crate) const CONTRADICTION_VETO: f32 = 0.8;

/// Maximum turns reranked per question (`SEMANTIC_RERANK_BUDGET`).
pub(crate) fn rerank_budget() -> usize {
    std::env::var("SEMANTIC_RERANK_BUDGET")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(48)
}

fn env_threshold(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| (0.0..=1.0).contains(value))
        .unwrap_or(default)
}

/// Grant threshold; other reranker models are calibrated differently, so
/// deployments override it with `SEMANTIC_RELEVANCE_GRANT`.
pub(crate) fn grant_threshold() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| env_threshold("SEMANTIC_RELEVANCE_GRANT", RELEVANCE_GRANT))
}

/// Revoke threshold (`SEMANTIC_RELEVANCE_REVOKE`).
pub(crate) fn revoke_threshold() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| env_threshold("SEMANTIC_RELEVANCE_REVOKE", RELEVANCE_REVOKE))
}

fn env_value(name: &str, default: f32) -> f32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<f32>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(default)
}

/// Extra turns reranked beyond the lexical budget, chosen by embedding
/// similarity (`SEMANTIC_RERANK_DENSE`). Off by default: on the LoCoMo dev
/// sample it raised gold-turn coverage (75% -> 86% at 48) but not answer
/// quality, while doubling rerank latency.
pub(crate) fn rerank_dense_budget() -> usize {
    std::env::var("SEMANTIC_RERANK_DENSE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0)
}

/// Turns scoring at least this much get sentence-level judgments
/// (`SEMANTIC_SENTENCE_RESCORE_FLOOR`).
pub(crate) fn sentence_rescore_floor() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| env_value("SEMANTIC_SENTENCE_RESCORE_FLOOR", 0.1))
}

/// Score offset separating confidently relevant, uncertain, and confidently
/// irrelevant evidence (`SEMANTIC_RELEVANCE_TIER`); lexical scores only order
/// atoms within a tier.
pub(crate) fn relevance_tier() -> f32 {
    static VALUE: OnceLock<f32> = OnceLock::new();
    *VALUE.get_or_init(|| env_value("SEMANTIC_RELEVANCE_TIER", 6.0))
}

/// The confident ends of the relevance scale decide; the uncertain middle
/// defers to the caller's other evidence.
pub(crate) fn relevance_decision(relevance: Option<f32>) -> Option<bool> {
    match relevance {
        Some(score) if score >= grant_threshold() => Some(true),
        Some(score) if score < revoke_threshold() => Some(false),
        _ => None,
    }
}

#[cfg_attr(test, allow(dead_code))]
static SCORER: OnceLock<Option<Arc<dyn SemanticScorer>>> = OnceLock::new();

#[cfg(test)]
thread_local! {
    static TEST_SCORER: std::cell::RefCell<Option<Arc<dyn SemanticScorer>>> =
        const { std::cell::RefCell::new(None) };
}

/// Installs a scorer for the current test thread; tests never load models.
#[cfg(test)]
pub(crate) fn set_test_scorer(scorer: Option<Arc<dyn SemanticScorer>>) {
    TEST_SCORER.with(|slot| *slot.borrow_mut() = scorer);
}

/// Scorer used by the answer pipeline. Never blocks: while the models are
/// still loading (see `start_background_load`) answers use lexical checks.
pub(crate) fn active_scorer() -> Option<Arc<dyn SemanticScorer>> {
    #[cfg(test)]
    {
        TEST_SCORER.with(|slot| slot.borrow().clone())
    }
    #[cfg(not(test))]
    {
        SCORER.get().cloned().flatten()
    }
}

/// Loads the configured semantic models on a background thread at startup,
/// so no request ever waits for (or blocks the async runtime on) model
/// loading.
pub(crate) fn start_background_load() {
    std::thread::Builder::new()
        .name("semantic-model-load".into())
        .spawn(|| {
            let started = std::time::Instant::now();
            let _ = global_scorer();
            let _ = crate::semantic_retrieval::global_index_blocking();
            eprintln!("semantic models ready in {:?}", started.elapsed());
        })
        .map(|_| ())
        .unwrap_or_else(|error| eprintln!("semantic model loader not started: {error}"));
}

/// `loading`, `ready`, or `off` for backend status reporting.
pub(crate) fn semantic_model_status() -> &'static str {
    match SCORER.get() {
        None => "loading",
        Some(Some(_)) => "ready",
        Some(None) => "off",
    }
}

/// Deterministic scorer for tests: relevance from a (question, passage)
/// substring table, no contradiction unless listed.
#[cfg(test)]
pub(crate) struct TableScorer {
    pub(crate) relevance: Vec<(&'static str, f32)>,
    pub(crate) contradictions: Vec<(&'static str, &'static str)>,
}

#[cfg(test)]
impl SemanticScorer for TableScorer {
    fn backend_id(&self) -> &str {
        "test:table"
    }

    fn relevance(&self, _question: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        Ok(passages
            .iter()
            .map(|passage| {
                // Plain entries match the judged line; `context:` entries match
                // anywhere, so a test can model relevance borrowed from the
                // preceding turn.
                let last_line = passage.lines().last().unwrap_or(passage);
                let contextual = self.relevance.iter().find(|(needle, _)| {
                    passage.contains('\n')
                        && needle
                            .strip_prefix("context:")
                            .is_some_and(|needle| passage.contains(needle))
                });
                contextual
                    .or_else(|| {
                        self.relevance.iter().find(|(needle, _)| {
                            !needle.starts_with("context:") && last_line.contains(needle)
                        })
                    })
                    .map(|(_, score)| *score)
                    .unwrap_or(0.2)
            })
            .collect())
    }

    fn entailment(&self, pairs: &[(&str, &str)]) -> Result<Vec<EntailmentScores>, String> {
        Ok(pairs
            .iter()
            .map(|(premise, hypothesis)| {
                let contradiction = self.contradictions.iter().any(|(source, claim)| {
                    premise.contains(source) && hypothesis.to_lowercase().contains(&claim.to_lowercase())
                });
                EntailmentScores {
                    entailment: if contradiction { 0.02 } else { 0.5 },
                    neutral: if contradiction { 0.03 } else { 0.45 },
                    contradiction: if contradiction { 0.95 } else { 0.05 },
                }
            })
            .collect())
    }
}

/// Process-wide scorer; `None` when disabled or unavailable (the pipeline
/// then keeps its lexical checks).
#[cfg_attr(test, allow(dead_code))]
pub(crate) fn global_scorer() -> Option<Arc<dyn SemanticScorer>> {
    SCORER
        .get_or_init(|| match scorer_from_env() {
            Ok(scorer) => {
                match &scorer {
                    Some(scorer) => eprintln!("semantic scorer loaded: {}", scorer.backend_id()),
                    None => eprintln!("semantic scorer disabled; lexical answer checks only"),
                }
                scorer
            }
            Err(error) => {
                eprintln!("semantic scorer unavailable: {error}");
                None
            }
        })
        .clone()
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_and_sigmoid_are_normalized() {
        let probabilities = softmax(&[2.0, 1.0, -1.0]);
        assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn score_cache_persists_scores_but_not_text() {
        let dir = std::env::temp_dir().join(format!("umem-score-cache-{}", std::process::id()));
        let path = dir.join("scores.bin");
        let _ = std::fs::remove_file(&path);
        let calls = std::cell::Cell::new(0);
        {
            let cache = ScoreCache::persistent(&path);
            let scores = cache
                .scores(0, &[("question", "a secret passage")], |pairs| {
                    calls.set(calls.get() + pairs.len());
                    Ok(vec![vec![0.75]])
                })
                .unwrap();
            assert_eq!(scores, vec![vec![0.75]]);
        }
        let reopened = ScoreCache::persistent(&path);
        let scores = reopened
            .scores(0, &[("question", "a secret passage")], |_| panic!("served from the file"))
            .unwrap();
        assert_eq!(scores, vec![vec![0.75]]);
        assert_eq!(calls.get(), 1);
        let bytes = std::fs::read(&path).unwrap();
        assert!(!String::from_utf8_lossy(&bytes).contains("secret"), "no text on disk");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rerank_and_nli_responses_parse_for_each_api() {
        let tei = serde_json::json!([{"index": 1, "score": 0.2}, {"index": 0, "score": 0.9}]);
        assert_eq!(parse_rerank_response(&tei, 2).unwrap(), vec![0.9, 0.2]);
        let cohere = serde_json::json!({"results": [{"index": 0, "relevance_score": 0.7}, {"index": 1, "relevance_score": 0.1}]});
        assert_eq!(parse_rerank_response(&cohere, 2).unwrap(), vec![0.7, 0.1]);
        let voyage = serde_json::json!({"data": [{"index": 0, "relevance_score": 0.4}]});
        assert_eq!(parse_rerank_response(&voyage, 1).unwrap(), vec![0.4]);
        assert!(parse_rerank_response(&voyage, 2).is_err(), "omitted passages fail closed");
        assert!(parse_rerank_response(&serde_json::json!([{"index": 5, "score": 0.1}]), 1).is_err());
        let nli = serde_json::json!([[
            {"label": "ENTAILMENT", "score": 0.1},
            {"label": "neutral", "score": 0.2},
            {"label": "contradiction", "score": 0.7}
        ]]);
        let parsed = parse_nli_response(&nli, 1).unwrap();
        assert!((parsed[0].contradiction - 0.7).abs() < 1e-6);
        assert!(parse_nli_response(&nli, 2).is_err());
        assert_eq!(
            rerank_payload(RerankApi::Cohere, "rerank-v3", "q", &["a"])["documents"],
            serde_json::json!(["a"])
        );
        assert_eq!(rerank_payload(RerankApi::Tei, "", "q", &["a"])["texts"], serde_json::json!(["a"]));
    }


    fn mock_http_server(body: &'static str) -> (String, std::thread::JoinHandle<String>) {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/rerank", listener.local_addr().unwrap());
        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                let read = stream.read(&mut buffer).unwrap();
                request.extend_from_slice(&buffer[..read]);
                let text = String::from_utf8_lossy(&request);
                if let Some(end) = text.find("\r\n\r\n") {
                    let length = text[..end]
                        .lines()
                        .find_map(|line| line.to_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                        .unwrap_or(0);
                    if request.len() >= end + 4 + length {
                        break;
                    }
                }
                if read == 0 {
                    break;
                }
            }
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            String::from_utf8_lossy(&request).to_string()
        });
        (url, handle)
    }

    fn http_scorer(url: String) -> HttpSemanticScorer {
        HttpSemanticScorer {
            id: "http:tei:test".into(),
            gateway: crate::provider_egress::ProviderEgressGateway::from_env(),
            rerank_url: url,
            rerank_api: RerankApi::Tei,
            rerank_model: String::new(),
            nli_url: None,
            api_key: None,
            timeout_ms: 5_000,
            cache: ScoreCache::new(),
        }
    }

    #[test]
    fn http_scorer_fails_closed_without_request_scope() {
        let scorer = http_scorer("http://127.0.0.1:9/rerank".into());
        assert!(scorer.remote());
        let error = scorer.relevance("q", &["passage"]).unwrap_err();
        assert!(error.contains("request_scope_missing"), "{error}");
        assert!(scorer.entailment(&[("a", "b")]).is_err(), "no NLI endpoint configured");
    }

    #[test]
    fn http_scorer_posts_through_egress_with_the_request_scope() {
        let (url, server) = mock_http_server(r#"[{"index":0,"score":0.83}]"#);
        let scorer = http_scorer(url);
        let scope = crate::embedding::EmbeddingScope::new("default", "default", false);
        let scores = with_request_scope(Some(scope), || scorer.relevance("When did James cook?", &["James: I cooked."]))
            .expect("mock rerank");
        assert_eq!(scores, vec![0.83]);
        let request = server.join().unwrap();
        assert!(request.starts_with("POST /rerank"), "{request}");
        assert!(request.contains("\"texts\":[\"James: I cooked.\"]"), "{request}");
        assert!(request_scope().is_none(), "scope is restored after the call");
        // Cached: a repeat call needs no server.
        let scope = crate::embedding::EmbeddingScope::new("default", "default", false);
        assert_eq!(
            with_request_scope(Some(scope), || scorer.relevance("When did James cook?", &["James: I cooked."])).unwrap(),
            vec![0.83]
        );
    }

    /// Session retrieval calibration on LoCoMo dev conversations only:
    /// gold session in the top 10 by lexical overlap, by best-turn embedding
    /// similarity, and by blends.
    /// `LOCOMO_DATASET=... cargo test --release --lib embedding_calibration -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn embedding_calibration() {
        const DEV: &[&str] = &["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"];
        let path = std::env::var("LOCOMO_DATASET").expect("LOCOMO_DATASET");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let embedder = OnnxEmbedder::load(
            &env_path("SEMANTIC_EMBEDDING_MODEL_DIR", "multilingual-e5-small"),
            6,
        )
        .expect("embedding model");
        let words = |text: &str| {
            text.to_lowercase()
                .split(|character: char| !character.is_alphanumeric())
                .filter(|word| word.len() > 2)
                .map(str::to_string)
                .collect::<std::collections::HashSet<_>>()
        };
        let weights = [0.0_f32, 0.25, 0.5, 0.75, 1.0];
        let mut hits = vec![0_usize; weights.len()];
        let mut total = 0_usize;
        let started = std::time::Instant::now();
        let mut embedded_turns = 0_usize;
        for sample in data.as_array().unwrap() {
            if !DEV.contains(&sample["sample_id"].as_str().unwrap()) {
                continue;
            }
            let conversation = sample["conversation"].as_object().unwrap();
            let mut sessions = Vec::<(usize, Vec<String>)>::new();
            for (key, value) in conversation {
                let Some(number) = key.strip_prefix("session_").and_then(|rest| rest.parse::<usize>().ok()) else {
                    continue;
                };
                let turns = value
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|turn| format!("{}: {}", turn["speaker"].as_str().unwrap_or(""), turn["text"].as_str().unwrap_or("")))
                    .collect::<Vec<_>>();
                sessions.push((number, turns));
            }
            let session_vectors = sessions
                .iter()
                .map(|(number, turns)| {
                    embedded_turns += turns.len();
                    let refs = turns.iter().map(String::as_str).collect::<Vec<_>>();
                    (*number, embedder.embed_passages(&refs).unwrap(), words(&turns.join(" ")))
                })
                .collect::<Vec<_>>();
            for qa in sample["qa"].as_array().unwrap() {
                if qa["category"].as_i64() == Some(5) {
                    continue;
                }
                let gold = qa["evidence"]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|value| value.as_str())
                            .filter_map(|id| id.trim_start_matches('D').split(':').next()?.parse::<usize>().ok())
                            .collect::<std::collections::HashSet<_>>()
                    })
                    .unwrap_or_default();
                if gold.is_empty() {
                    continue;
                }
                let question = qa["question"].as_str().unwrap();
                let query = embedder.embed_queries(&[question]).unwrap().remove(0);
                let query_words = words(question);
                let lexical = session_vectors
                    .iter()
                    .map(|(_, _, session_words)| {
                        query_words.intersection(session_words).count() as f32 / (query_words.len().max(1) as f32)
                    })
                    .collect::<Vec<_>>();
                let semantic = session_vectors
                    .iter()
                    .map(|(_, vectors, _)| vectors.iter().map(|vector| dot(&query, vector)).fold(f32::MIN, f32::max))
                    .collect::<Vec<_>>();
                total += 1;
                for (index, weight) in weights.iter().enumerate() {
                    let mut ranked = session_vectors
                        .iter()
                        .enumerate()
                        .map(|(position, (number, _, _))| (*number, (1.0 - weight) * lexical[position] + weight * semantic[position]))
                        .collect::<Vec<_>>();
                    ranked.sort_by(|left, right| right.1.partial_cmp(&left.1).unwrap());
                    if ranked.iter().take(10).any(|(number, _)| gold.contains(number)) {
                        hits[index] += 1;
                    }
                }
            }
        }
        eprintln!("{total} dev questions; embedded {embedded_turns} turns in {:?}", started.elapsed());
        for (weight, hit) in weights.iter().zip(&hits) {
            eprintln!("  semantic weight {weight:.2}: gold session in top 10 = {:.1}%", 100.0 * *hit as f32 / total as f32);
        }
    }

    /// Standalone-sentence calibration (dev only): best standalone score of
    /// gold-turn sentences for answerable vs adversarial questions.
    #[test]
    #[ignore]
    fn standalone_calibration() {
        const DEV: &[&str] = &["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"];
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(std::env::var("LOCOMO_DATASET").unwrap()).unwrap()).unwrap();
        let scorer = OnnxSemanticScorer::load(
            &env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-v2-m3"),
            &env_path("SEMANTIC_NLI_MODEL_DIR", "mdeberta-v3-base-xnli"),
            4,
        )
        .unwrap();
        let limit = 160;
        let mut answerable = Vec::new();
        let mut adversarial = Vec::new();
        for sample in data.as_array().unwrap() {
            if !DEV.contains(&sample["sample_id"].as_str().unwrap()) {
                continue;
            }
            let conversation = sample["conversation"].as_object().unwrap();
            let mut turns = HashMap::new();
            for (key, value) in conversation {
                if key.starts_with("session_") && !key.ends_with("date_time") {
                    for turn in value.as_array().unwrap() {
                        turns.insert(turn["dia_id"].as_str().unwrap().to_string(), (turn["speaker"].as_str().unwrap().to_string(), turn["text"].as_str().unwrap().to_string()));
                    }
                }
            }
            for qa in sample["qa"].as_array().unwrap() {
                let adversarial_question = qa["category"].as_i64() == Some(5);
                let bucket = if adversarial_question { &mut adversarial } else { &mut answerable };
                if bucket.len() >= limit {
                    continue;
                }
                let sentences = qa["evidence"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|id| turns.get(id.as_str()?))
                    .flat_map(|(speaker, text)| {
                        text.split_inclusive(['.', '!', '?'])
                            .map(str::trim)
                            .filter(|sentence| sentence.split_whitespace().count() >= 2)
                            .map(move |sentence| format!("{speaker}: {sentence}"))
                    })
                    .collect::<Vec<_>>();
                if sentences.is_empty() {
                    continue;
                }
                let question = qa["question"].as_str().unwrap();
                let scores = scorer
                    .relevance(question, &sentences.iter().map(String::as_str).collect::<Vec<_>>())
                    .unwrap();
                bucket.push(scores.into_iter().fold(0.0_f32, f32::max));
            }
        }
        for (name, values) in [("answerable", &mut answerable), ("adversarial", &mut adversarial)] {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let pct = |p: f32| values[((values.len() as f32 - 1.0) * p) as usize];
            eprintln!("{name}: n={} p10={:.3} p25={:.3} p50={:.3} p75={:.3}", values.len(), pct(0.10), pct(0.25), pct(0.5), pct(0.75));
            for bar in [0.01_f32, 0.03, 0.05, 0.1, 0.2] {
                eprintln!("   >= {bar}: {:.1}%", 100.0 * values.iter().filter(|v| **v >= bar).count() as f32 / values.len() as f32);
            }
        }
    }

    /// Threshold calibration on LoCoMo dev conversations only (26/30/41/42/43):
    /// gold evidence turns vs. the 20 most word-similar other turns.
    /// `LOCOMO_DATASET=... cargo test --release --lib semantic_calibration -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn semantic_calibration() {
        const DEV: &[&str] = &["conv-26", "conv-30", "conv-41", "conv-42", "conv-43"];
        let limit = std::env::var("CALIBRATION_QUESTIONS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(250);
        let path = std::env::var("LOCOMO_DATASET").expect("LOCOMO_DATASET");
        let data: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let scorer = OnnxSemanticScorer::load(
            &env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-base"),
            &env_path("SEMANTIC_NLI_MODEL_DIR", "mdeberta-v3-base-xnli"),
            6,
        )
        .expect("models");
        let only_context = std::env::var("CALIBRATION_CONTEXT_ONLY").is_ok();
        let words = |text: &str| {
            text.to_lowercase()
                .split(|character: char| !character.is_alphanumeric())
                .filter(|word| word.len() > 3)
                .map(str::to_string)
                .collect::<std::collections::HashSet<_>>()
        };
        let mut rows = Vec::new(); // (variant, gold_best, negative_best, gold_rank)
        let mut asked = 0;
        'samples: for sample in data.as_array().unwrap() {
            let id = sample["sample_id"].as_str().unwrap();
            if !DEV.contains(&id) {
                continue;
            }
            let conversation = sample["conversation"].as_object().unwrap();
            let mut turns = Vec::<(String, String, String)>::new(); // id, plain, with context
            let mut sessions = conversation
                .keys()
                .filter(|key| key.starts_with("session_") && !key.ends_with("date_time"))
                .filter_map(|key| key[8..].parse::<usize>().ok().map(|n| (n, key.clone())))
                .collect::<Vec<_>>();
            sessions.sort();
            for (_, key) in sessions {
                let mut previous: Option<String> = None;
                for turn in conversation[&key].as_array().unwrap() {
                    let line = format!(
                        "{}: {}",
                        turn["speaker"].as_str().unwrap_or(""),
                        turn["text"].as_str().unwrap_or("")
                    );
                    let with_context = match &previous {
                        Some(previous) => format!("{previous}\n{line}"),
                        None => line.clone(),
                    };
                    turns.push((
                        turn["dia_id"].as_str().unwrap_or("").to_string(),
                        line.clone(),
                        with_context,
                    ));
                    previous = Some(line);
                }
            }
            for qa in sample["qa"].as_array().unwrap() {
                if qa["category"].as_i64() == Some(5) {
                    continue;
                }
                let question = qa["question"].as_str().unwrap();
                let gold = qa["evidence"]
                    .as_array()
                    .map(|values| {
                        values
                            .iter()
                            .filter_map(|value| value.as_str().map(str::to_string))
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let gold_turns = turns
                    .iter()
                    .filter(|turn| gold.contains(&turn.0))
                    .collect::<Vec<_>>();
                if gold_turns.is_empty() {
                    continue;
                }
                let category = qa["category"].as_i64().unwrap_or(0);
                let question_words = words(question);
                let lexical_gold = gold_turns
                    .iter()
                    .map(|turn| words(&turn.1).intersection(&question_words).count())
                    .max()
                    .unwrap_or(0);
                let mut negatives = turns
                    .iter()
                    .filter(|turn| !gold.contains(&turn.0))
                    .map(|turn| (words(&turn.1).intersection(&question_words).count(), turn))
                    .collect::<Vec<_>>();
                negatives.sort_by(|left, right| right.0.cmp(&left.0));
                let negatives = negatives.into_iter().take(20).collect::<Vec<_>>();
                let lexical_rank = negatives.iter().filter(|(overlap, _)| *overlap > lexical_gold).count();
                rows.push((2, lexical_gold as f32, 0.0, lexical_rank, Vec::new(), category));
                let negatives = negatives.into_iter().map(|(_, turn)| turn).collect::<Vec<_>>();
                for variant in if only_context { 1..2 } else { 0..2 } {
                    let text = |turn: &(String, String, String)| {
                        if variant == 0 { turn.1.clone() } else { turn.2.clone() }
                    };
                    let gold_texts = gold_turns.iter().map(|turn| text(turn)).collect::<Vec<_>>();
                    let negative_texts = negatives.iter().map(|turn| text(turn)).collect::<Vec<_>>();
                    let all = gold_texts
                        .iter()
                        .chain(&negative_texts)
                        .map(String::as_str)
                        .collect::<Vec<_>>();
                    let scores = scorer.relevance(question, &all).unwrap();
                    let (gold_scores, negative_scores) = scores.split_at(gold_texts.len());
                    let gold_best = gold_scores.iter().copied().fold(0.0, f32::max);
                    let negative_best = negative_scores.iter().copied().fold(0.0, f32::max);
                    let rank = negative_scores.iter().filter(|score| **score > gold_best).count();
                    rows.push((variant, gold_best, negative_best, rank, negative_scores.to_vec(), category));
                }
                asked += 1;
                if asked >= limit {
                    break 'samples;
                }
            }
        }
        let reranker = env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-base");
        eprintln!("reranker {}", reranker.display());
        for variant in [2, 0, 1] {
            let all_rows = rows.iter().filter(|row| row.0 == variant).collect::<Vec<_>>();
            eprintln!(
                "variant {} ({} questions)",
                ["plain", "with previous turn", "word-overlap baseline"][variant],
                all_rows.len()
            );
            for category in [1_i64, 2, 3, 4] {
                let rows = all_rows.iter().filter(|row| row.5 == category).collect::<Vec<_>>();
                if !rows.is_empty() {
                    eprintln!(
                        "  category {category}: n={} first {:.1}% top-3 {:.1}%",
                        rows.len(),
                        100.0 * rows.iter().filter(|row| row.3 == 0).count() as f32 / rows.len() as f32,
                        100.0 * rows.iter().filter(|row| row.3 < 3).count() as f32 / rows.len() as f32
                    );
                }
            }
            let rows = all_rows;
            let count = rows.len() as f32;
            eprintln!(
                "  gold ranked first: {:.1}%  top-3: {:.1}%",
                100.0 * rows.iter().filter(|row| row.3 == 0).count() as f32 / count,
                100.0 * rows.iter().filter(|row| row.3 < 3).count() as f32 / count
            );
            if variant == 2 {
                continue;
            }
            for threshold in [0.01_f32, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5] {
                let gold_recall = rows.iter().filter(|row| row.1 >= threshold).count() as f32 / count;
                let negatives = rows.iter().flat_map(|row| row.4.iter()).collect::<Vec<_>>();
                let false_accept = negatives.iter().filter(|score| ***score >= threshold).count() as f32
                    / negatives.len().max(1) as f32;
                eprintln!(
                    "  t={threshold:<5} gold recall {:.1}%  hard-negative accept {:.1}%",
                    100.0 * gold_recall,
                    100.0 * false_accept
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn model_load_timing() {
        for (name, level) in [("default", true)] {
            let started = std::time::Instant::now();
            let _scorer = OnnxSemanticScorer::load(
                &env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-v2-m3"),
                &env_path("SEMANTIC_NLI_MODEL_DIR", "mdeberta-v3-base-xnli"),
                6,
            )
            .unwrap();
            eprintln!("TIMING {name} {level} scorer load {:?}", started.elapsed());
        }
        let started = std::time::Instant::now();
        let _embedder = OnnxEmbedder::load(&env_path("SEMANTIC_EMBEDDING_MODEL_DIR", "multilingual-e5-small"), 6).unwrap();
        eprintln!("TIMING embedder load {:?}", started.elapsed());
    }

    #[test]
    #[ignore]
    fn passage_format_probe() {
        let scorer = OnnxSemanticScorer::load(
            &env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-v2-m3"),
            &env_path("SEMANTIC_NLI_MODEL_DIR", "mdeberta-v3-base-xnli"),
            6,
        )
        .unwrap();
        let question = "What type of individuals does the adoption agency Caroline is considering support?";
        let variants = [
            "I chose them 'cause they help LGBTQ+ folks with adoption.",
            "Caroline: I chose them 'cause they help LGBTQ+ folks with adoption.",
            "Melanie: Wow, that agency looks great! What made you pick it?\nCaroline: I chose them 'cause they help LGBTQ+ folks with adoption.",
            "Melanie: Wow, that agency looks great! What made you pick it?\nCaroline: I chose them 'cause they help LGBTQ+ folks with adoption. Their inclusivity and support really spoke to me.",
            "Caroline said: the adoption agency I chose helps LGBTQ+ folks with adoption.",
        ];
        for (variant, score) in variants.iter().zip(scorer.relevance(question, &variants).unwrap()) {
            eprintln!("PROBE {score:.3}  {}", variant.replace('\n', " / "));
        }
    }

    /// Manual probe against the downloaded models:
    /// `cargo test --lib semantic_probe -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn semantic_probe() {
        let scorer = OnnxSemanticScorer::load(
            &env_path("SEMANTIC_RERANKER_MODEL_DIR", "bge-reranker-base"),
            &env_path("SEMANTIC_NLI_MODEL_DIR", "mdeberta-v3-base-xnli"),
            4,
        )
        .expect("models");
        let question = "When did James start taking cooking classes?";
        let passages = [
            "James: Yes, two days ago I signed up for a cooking class. At the first lesson we prepared several simple dishes.",
            "James: Yesterday, we started on a road trip. Exploring new places and taking in nature was awesome!",
            "John: Wow, Calvin!",
            "James: I got a great omelette the first time!",
        ];
        let started = std::time::Instant::now();
        let relevance = scorer.relevance(question, &passages).unwrap();
        eprintln!("relevance in {:?}", started.elapsed());
        for (passage, score) in passages.iter().zip(&relevance) {
            eprintln!("  {score:.3}  {passage}");
        }
        let pairs = [
            (passages[0], "James signed up for a cooking class."),
            (passages[0], "James started taking cooking classes."),
            (passages[1], "James started taking cooking classes."),
            (passages[1], "James started a road trip."),
            ("James: I never liked cooking.", "James loves cooking."),
            ("Tim: Definitely Star Wars! It is my favorite.", "Tim likes Star Wars."),
        ];
        let started = std::time::Instant::now();
        let entailment = scorer.entailment(&pairs).unwrap();
        eprintln!("entailment in {:?}", started.elapsed());
        for ((premise, hypothesis), score) in pairs.iter().zip(&entailment) {
            eprintln!(
                "  e={:.3} n={:.3} c={:.3}  {premise} => {hypothesis}",
                score.entailment, score.neutral, score.contradiction
            );
        }
    }
}
