//! Optional cross-encoder reranking (e.g. BAAI/bge-reranker-base or
//! bge-reranker-v2-m3 exported to ONNX), run locally by ONNX Runtime.
//!
//! `TIDEMARK_RERANK_DIR` names a directory with `model.onnx` and
//! `tokenizer.json`; `TIDEMARK_RERANK_CANDIDATES` (default 0 = off) is how many
//! fused candidates are re-ordered by the cross-encoder.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use tokenizers::{Tokenizer, TruncationParams, TruncationStrategy};

/// Tokens per query–record pair (`TIDEMARK_RERANK_MAX_TOKENS`); 512 is the
/// limit of BERT-size cross-encoders such as bge-reranker-base.
const DEFAULT_MAX_TOKENS: usize = 512;
const BATCH_SIZE: usize = 16;

/// Pairs scored and pairs cut at the token limit since start (their text
/// beyond the limit did not reach the model).
pub static PAIRS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static TRUNCATED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Opens an ONNX model. Full graph optimization of a base-size model takes
/// over a minute on a small CPU, so the optimized graph is saved next to the
/// model (`model.optimized.onnx`) on first load and reused afterwards; a
/// read-only directory falls back to optimizing in memory.
fn load_session(model: &Path, threads: usize) -> Result<Session, String> {
    let optimized = model.with_file_name("model.optimized.onnx");
    let modified = |path: &Path| std::fs::metadata(path).and_then(|meta| meta.modified()).ok();
    let fresh = matches!((modified(&optimized), modified(model)), (Some(cache), Some(source)) if cache >= source);
    let builder = |level: GraphOptimizationLevel| -> Result<ort::session::builder::SessionBuilder, String> {
        Session::builder()
            .map_err(|e| e.to_string())?
            .with_optimization_level(level)
            .map_err(|e| e.to_string())?
            .with_intra_threads(threads)
            .map_err(|e| e.to_string())
    };
    if fresh {
        if let Ok(session) = builder(GraphOptimizationLevel::Disable)?.commit_from_file(&optimized) {
            return Ok(session);
        }
    }
    let saved = builder(GraphOptimizationLevel::Level3)?
        .with_optimized_model_path(&optimized)
        .map_err(|e| e.to_string())
        .and_then(|mut b| b.commit_from_file(model).map_err(|e| e.to_string()));
    match saved {
        Ok(session) => Ok(session),
        Err(_) => builder(GraphOptimizationLevel::Level3)?
            .commit_from_file(model)
            .map_err(|e| format!("{}: {e}", model.display())),
    }
}

pub struct Reranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    pub label: String,
}

impl Reranker {
    pub fn from_env() -> Result<Option<Self>, String> {
        let Some(dir) = std::env::var("TIDEMARK_RERANK_DIR").ok().map(|d| d.trim().to_string()).filter(|d| !d.is_empty()) else {
            return Ok(None);
        };
        let threads = std::env::var("TIDEMARK_RERANK_THREADS").ok().and_then(|v| v.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
        let max_tokens = std::env::var("TIDEMARK_RERANK_MAX_TOKENS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(DEFAULT_MAX_TOKENS).max(16);
        Self::load(Path::new(&dir), threads, max_tokens).map(Some)
    }

    pub fn load(dir: &Path, threads: usize, max_tokens: usize) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_tokens,
                strategy: TruncationStrategy::LongestFirst,
                ..TruncationParams::default()
            }))
            .map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        let session = load_session(&dir.join("model.onnx"), threads)?;
        let input_names = session.inputs().iter().map(|input| input.name().to_string()).collect();
        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            input_names,
            label: dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
        })
    }

    /// Relevance score of each passage for the query (higher = more relevant).
    pub fn scores(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        let mut results = Vec::with_capacity(passages.len());
        let pad_id = self.tokenizer.token_to_id("<pad>").or_else(|| self.tokenizer.token_to_id("[PAD]")).map(i64::from).unwrap_or(0);
        for chunk in passages.chunks(BATCH_SIZE) {
            let encodings = self
                .tokenizer
                .encode_batch(chunk.iter().map(|p| (query.to_string(), p.to_string())).collect::<Vec<_>>(), true)
                .map_err(|e| e.to_string())?;
            let cut = encodings.iter().filter(|e| !e.get_overflowing().is_empty()).count() as u64;
            PAIRS.fetch_add(encodings.len() as u64, std::sync::atomic::Ordering::Relaxed);
            TRUNCATED.fetch_add(cut, std::sync::atomic::Ordering::Relaxed);
            let width = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);
            let rows = encodings.len();
            let mut ids = vec![pad_id; rows * width];
            let mut mask = vec![0_i64; rows * width];
            let mut types = vec![0_i64; rows * width];
            for (row, encoding) in encodings.iter().enumerate() {
                let offset = row * width;
                for (column, ((id, attention), type_id)) in
                    encoding.get_ids().iter().zip(encoding.get_attention_mask()).zip(encoding.get_type_ids()).enumerate()
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
                let tensor = Tensor::from_array((shape, data)).map_err(|e| e.to_string())?;
                inputs.push((name.clone(), tensor.into_dyn()));
            }
            let mut session = self.session.lock().map_err(|_| "reranker session poisoned".to_string())?;
            let outputs = session.run(inputs).map_err(|e| e.to_string())?;
            let (out_shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
            let labels = out_shape.last().copied().unwrap_or(1).max(1) as usize;
            if values.len() != rows * labels {
                return Err(format!("unexpected output size {} for {rows} rows x {labels} labels", values.len()));
            }
            // A single-logit reranker: the logit is the relevance score.
            results.extend(values.chunks(labels).map(|row| row[labels - 1]));
        }
        Ok(results)
    }
}
