//! Optional reranking of the leading search candidates
//! (`TIDEMARK_RERANK_CANDIDATES`, default 0 = off) by a cross-encoder: run
//! locally by ONNX Runtime, or by a reranking API.
//!
//! Local: `TIDEMARK_RERANK_DIR` names a directory with `model.onnx` and
//! `tokenizer.json` (e.g. BAAI/bge-reranker-base or bge-reranker-v2-m3 exported to ONNX).
//!
//! API: `TIDEMARK_RERANK_PROTOCOL` = cohere (also vLLM, SiliconFlow, Xinference …)
//! | jina | voyage | tei | dashscope | template, with `TIDEMARK_RERANK_URL`
//! (hosted providers have a default), `_API_KEY`, `_MODEL`, `_HEADERS`, `_AUTH`
//! (bearer | none | header:Name), `_BATCH` (documents per request), `_MAX_CHARS`
//! (characters of a record sent, default 8000), and for the template `_TEMPLATE`
//! (JSON body with "{{query}}", "{{documents}}", {{model}}, {{top_n}}),
//! `_TEMPLATE_RESULTS` (path to the result list, default "results"),
//! `_TEMPLATE_INDEX` ("index") and `_TEMPLATE_SCORE` ("relevance_score").
//! A result list of plain numbers is read as scores in document order.

use crate::http::{self, Auth};
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use std::sync::Mutex;
use serde_json::{json, Value};
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
pub(crate) fn load_session(model: &Path, threads: usize) -> Result<Session, String> {
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

/// Units the reranking API reported since start (tokens, or Cohere's search units).
pub static API_UNITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub enum Reranker {
    Local(Onnx),
    Remote(Remote),
}

impl Reranker {
    /// A reranking API if `TIDEMARK_RERANK_PROTOCOL` is set, else a local model
    /// if `TIDEMARK_RERANK_DIR` is set, else none.
    pub fn from_env() -> Result<Option<Self>, String> {
        if let Some(remote) = Remote::from_lookup(|k| std::env::var(k).ok())? {
            return Ok(Some(Reranker::Remote(remote)));
        }
        Ok(Onnx::from_env()?.map(Reranker::Local))
    }

    pub fn label(&self) -> String {
        match self {
            Reranker::Local(o) => o.label.clone(),
            Reranker::Remote(r) => format!("{}:{}", r.protocol.name(), r.model),
        }
    }

    /// Relevance score of each passage for the query (higher = more relevant).
    /// Called from blocking code (search ranks inside `block_in_place`); an API
    /// call waits on the runtime there.
    pub fn scores(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        match self {
            Reranker::Local(o) => o.scores(query, passages),
            Reranker::Remote(r) => tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(r.scores(query, passages))),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Cohere,
    Jina,
    Voyage,
    Tei,
    DashScope,
    Template,
}

impl Protocol {
    fn parse(name: &str) -> Option<Protocol> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "cohere" | "vllm" | "siliconflow" | "xinference" => Protocol::Cohere,
            "jina" => Protocol::Jina,
            "voyage" | "voyageai" => Protocol::Voyage,
            "tei" | "huggingface-tei" => Protocol::Tei,
            "dashscope" | "qwen" => Protocol::DashScope,
            "template" | "custom" => Protocol::Template,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Protocol::Cohere => "cohere",
            Protocol::Jina => "jina",
            Protocol::Voyage => "voyage",
            Protocol::Tei => "tei",
            Protocol::DashScope => "dashscope",
            Protocol::Template => "template",
        }
    }

    fn default_url(self) -> Option<&'static str> {
        match self {
            Protocol::Cohere => Some("https://api.cohere.com/v2"),
            Protocol::Jina => Some("https://api.jina.ai/v1"),
            Protocol::Voyage => Some("https://api.voyageai.com/v1"),
            Protocol::DashScope => Some("https://dashscope.aliyuncs.com/api/v1"),
            Protocol::Tei | Protocol::Template => None,
        }
    }

    fn default_batch(self) -> usize {
        match self {
            Protocol::Tei => 32,
            Protocol::DashScope => 500,
            _ => 1000,
        }
    }
}

pub struct Remote {
    protocol: Protocol,
    url: String,
    key: Option<String>,
    model: String,
    headers: Vec<(String, String)>,
    auth: Auth,
    batch: usize,
    max_chars: usize,
    template: Option<Value>,
    template_results: String,
    template_index: String,
    template_score: String,
    client: reqwest::Client,
}

impl Remote {
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Option<Remote>, String> {
        let var = |k: &str| http::setting(&get, &format!("TIDEMARK_RERANK_{k}"));
        let Some(name) = var("PROTOCOL") else { return Ok(None) };
        let protocol = Protocol::parse(&name).ok_or_else(|| format!("unknown TIDEMARK_RERANK_PROTOCOL {name:?}"))?;
        let url = var("URL")
            .or_else(|| protocol.default_url().map(String::from))
            .ok_or_else(|| format!("TIDEMARK_RERANK_URL is required for the {} protocol", protocol.name()))?;
        let model = match (var("MODEL"), protocol) {
            (Some(m), _) => m,
            (None, Protocol::Tei | Protocol::Template) => String::new(),
            (None, p) => return Err(format!("TIDEMARK_RERANK_MODEL is required for the {} protocol", p.name())),
        };
        let template = match var("TEMPLATE") {
            Some(t) => Some(serde_json::from_str::<Value>(&t).map_err(|e| format!("TIDEMARK_RERANK_TEMPLATE is not JSON: {e}"))?),
            None if protocol == Protocol::Template => return Err("the template protocol needs TIDEMARK_RERANK_TEMPLATE".into()),
            None => None,
        };
        Ok(Some(Remote {
            protocol,
            url: url.trim_end_matches('/').to_string(),
            key: var("API_KEY"),
            model,
            headers: var("HEADERS").map(|h| http::parse_headers(&h)).unwrap_or_default(),
            auth: Auth::parse(var("AUTH").as_deref(), "TIDEMARK_RERANK_AUTH").unwrap_or(Auth::Bearer),
            batch: var("BATCH").and_then(|v| v.parse().ok()).unwrap_or(protocol.default_batch()).max(1),
            max_chars: var("MAX_CHARS").and_then(|v| v.parse().ok()).unwrap_or(8000).max(1),
            template,
            template_results: var("TEMPLATE_RESULTS").unwrap_or_else(|| "results".into()),
            template_index: var("TEMPLATE_INDEX").unwrap_or_else(|| "index".into()),
            template_score: var("TEMPLATE_SCORE").unwrap_or_else(|| "relevance_score".into()),
            client: http::client(),
        }))
    }

    /// URL and JSON body for one batch of documents.
    fn request(&self, query: &str, documents: &[String]) -> (String, Value) {
        let n = documents.len();
        let endpoint = |path: &str| if self.url.ends_with(path) { self.url.clone() } else { format!("{}{path}", self.url) };
        match self.protocol {
            Protocol::Cohere => (endpoint("/rerank"), json!({"model": self.model, "query": query, "documents": documents, "top_n": n})),
            Protocol::Jina => (endpoint("/rerank"), json!({"model": self.model, "query": query, "documents": documents, "top_n": n, "return_documents": false})),
            Protocol::Voyage => (endpoint("/rerank"), json!({"model": self.model, "query": query, "documents": documents, "top_k": n, "truncation": true})),
            Protocol::Tei => (endpoint("/rerank"), json!({"query": query, "texts": documents, "truncate": true})),
            Protocol::DashScope => (
                endpoint("/services/rerank/text-rerank/text-rerank"),
                json!({"model": self.model, "input": {"query": query, "documents": documents}, "parameters": {"top_n": n, "return_documents": false}}),
            ),
            Protocol::Template => {
                let values = [("query", json!(query)), ("documents", json!(documents)), ("model", json!(self.model)), ("top_n", json!(n))];
                (self.url.clone(), http::fill(self.template.as_ref().expect("template"), &values))
            }
        }
    }

    /// Scores in document order, and the units the provider reported.
    fn parse(&self, value: &Value, n: usize) -> Result<(Vec<f32>, u64), String> {
        let (path, index_key, score_key) = match self.protocol {
            Protocol::Cohere | Protocol::Jina => ("results", "index", "relevance_score"),
            Protocol::Voyage => ("data", "index", "relevance_score"),
            Protocol::Tei => ("", "index", "score"),
            Protocol::DashScope => ("output.results", "index", "relevance_score"),
            Protocol::Template => (self.template_results.as_str(), self.template_index.as_str(), self.template_score.as_str()),
        };
        let items = http::select(value, path);
        let items: Vec<&Value> = match items.as_slice() {
            [Value::Array(a)] => a.iter().collect(),
            _ => items,
        };
        if items.is_empty() && n > 0 {
            return Err("rerank reply without results".into());
        }
        let mut scores = vec![None; n];
        for (position, item) in items.iter().enumerate() {
            let (index, score) = match item.as_f64() {
                Some(s) => (position, s),
                None => (
                    item[index_key].as_u64().ok_or("rerank result without index")? as usize,
                    item[score_key].as_f64().ok_or("rerank result without score")?,
                ),
            };
            *scores.get_mut(index).ok_or_else(|| format!("rerank result index {index} out of range"))? = Some(score as f32);
        }
        // Documents the provider left out rank below every scored one.
        let floor = scores.iter().flatten().fold(f32::INFINITY, |a, &b| a.min(b));
        let floor = if floor.is_finite() { floor - 1.0 } else { 0.0 };
        let units = value["meta"]["billed_units"]["search_units"]
            .as_u64()
            .or_else(|| value["usage"]["total_tokens"].as_u64())
            .unwrap_or(0);
        Ok((scores.into_iter().map(|s| s.unwrap_or(floor)).collect(), units))
    }

    pub async fn scores(&self, query: &str, passages: &[&str]) -> Result<Vec<f32>, String> {
        let mut cut = 0u64;
        let documents = passages
            .iter()
            .map(|p| {
                if p.chars().count() > self.max_chars {
                    cut += 1;
                    p.chars().take(self.max_chars).collect()
                } else {
                    p.to_string()
                }
            })
            .collect::<Vec<String>>();
        PAIRS.fetch_add(documents.len() as u64, std::sync::atomic::Ordering::Relaxed);
        TRUNCATED.fetch_add(cut, std::sync::atomic::Ordering::Relaxed);
        let headers = http::headers(self.key.as_deref(), &self.auth, &self.headers);
        let mut out = Vec::with_capacity(documents.len());
        for chunk in documents.chunks(self.batch) {
            let (url, body) = self.request(query, chunk);
            let value = http::post(&self.client, &url, &headers, &body, "rerank").await?;
            let (scores, units) = self.parse(&value, chunk.len())?;
            API_UNITS.fetch_add(units, std::sync::atomic::Ordering::Relaxed);
            out.extend(scores);
        }
        Ok(out)
    }
}

pub struct Onnx {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    pub label: String,
}

impl Onnx {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex as StdMutex};

    type Seen = Arc<StdMutex<Vec<(String, HashMap<String, String>, Value)>>>;

    /// A local server answering like `protocol`'s provider: document i scores
    /// its length (so longer documents rank first), results sorted by score as
    /// real APIs do. Records path, headers and body.
    async fn mock(protocol: Protocol) -> (String, Seen) {
        let seen: Seen = Arc::default();
        let log = seen.clone();
        let app = axum::Router::new().fallback(move |uri: axum::http::Uri, headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
            let log = log.clone();
            async move {
                let docs = ["documents", "texts", "docs"].iter().find_map(|k| body[k].as_array().cloned())
                    .or_else(|| body["input"]["documents"].as_array().cloned()).unwrap_or_default();
                let mut scored = docs.iter().enumerate().map(|(i, d)| (i, d.as_str().unwrap_or("").len() as f64)).collect::<Vec<_>>();
                scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let items = |score: &str| scored.iter().map(|(i, s)| json!({"index": i, score: s})).collect::<Vec<_>>();
                let reply = match protocol {
                    Protocol::Cohere => json!({"results": items("relevance_score"), "meta": {"billed_units": {"search_units": 1}}}),
                    Protocol::Jina => json!({"results": items("relevance_score"), "usage": {"total_tokens": 9}}),
                    Protocol::Voyage => json!({"data": items("relevance_score"), "usage": {"total_tokens": 9}}),
                    Protocol::Tei => json!(items("score")),
                    Protocol::DashScope => json!({"output": {"results": items("relevance_score")}, "usage": {"total_tokens": 9}}),
                    Protocol::Template => json!({"out": {"ranked": scored.iter().map(|(i, s)| json!({"doc": i, "value": s})).collect::<Vec<_>>()}}),
                };
                let h = headers.iter().map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())).collect();
                log.lock().unwrap().push((uri.to_string(), h, body));
                axum::Json(reply)
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, seen)
    }

    fn remote(vars: &[(&str, &str)]) -> Remote {
        let map = vars.iter().map(|(k, v)| (format!("TIDEMARK_RERANK_{k}"), v.to_string())).collect::<HashMap<_, _>>();
        Remote::from_lookup(|k| map.get(k).cloned()).unwrap().expect("configured")
    }

    #[tokio::test]
    async fn every_protocol_returns_scores_in_document_order() {
        let docs = ["aa", "a", "aaaa", "aaa"];
        for protocol in [Protocol::Cohere, Protocol::Jina, Protocol::Voyage, Protocol::Tei, Protocol::DashScope, Protocol::Template] {
            let (url, seen) = mock(protocol).await;
            let r = remote(&[
                ("PROTOCOL", protocol.name()),
                ("URL", &url),
                ("API_KEY", "secret"),
                ("MODEL", "rr1"),
                ("BATCH", "3"),
                ("TEMPLATE", r#"{"q": "{{query}}", "docs": "{{documents}}", "m": "{{model}}", "k": "{{top_n}}"}"#),
                ("TEMPLATE_RESULTS", "out.ranked"),
                ("TEMPLATE_INDEX", "doc"),
                ("TEMPLATE_SCORE", "value"),
            ]);
            let scores = r.scores("question", &docs).await.unwrap_or_else(|e| panic!("{protocol:?}: {e}"));
            assert_eq!(scores, vec![2.0, 1.0, 4.0, 3.0], "{protocol:?}");
            let log = seen.lock().unwrap().clone();
            assert_eq!(log.len(), 2, "{protocol:?}: batches of 3");
            assert_eq!(log[0].1["authorization"], "Bearer secret", "{protocol:?}");
        }
    }

    #[tokio::test]
    async fn requests_follow_each_providers_format() {
        let first = |seen: &Seen| seen.lock().unwrap()[0].clone();
        let (url, seen) = mock(Protocol::Cohere).await;
        remote(&[("PROTOCOL", "cohere"), ("URL", &url), ("MODEL", "rerank-v3.5")]).scores("q", &["d1", "d2"]).await.unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!(path, "/rerank");
        assert!(!headers.contains_key("authorization"), "no key, no auth header");
        assert_eq!(body, json!({"model": "rerank-v3.5", "query": "q", "documents": ["d1", "d2"], "top_n": 2}));

        let (url, seen) = mock(Protocol::Voyage).await;
        remote(&[("PROTOCOL", "voyage"), ("URL", &url), ("MODEL", "rerank-2.5")]).scores("q", &["d1"]).await.unwrap();
        assert_eq!(first(&seen).2, json!({"model": "rerank-2.5", "query": "q", "documents": ["d1"], "top_k": 1, "truncation": true}));

        let (url, seen) = mock(Protocol::Tei).await;
        remote(&[("PROTOCOL", "tei"), ("URL", &url), ("MAX_CHARS", "2"), ("HEADERS", "X-A: 1")]).scores("q", &["long text"]).await.unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!((path.as_str(), headers["x-a"].as_str()), ("/rerank", "1"));
        assert_eq!(body, json!({"query": "q", "texts": ["lo"], "truncate": true}));

        let (url, seen) = mock(Protocol::DashScope).await;
        remote(&[("PROTOCOL", "dashscope"), ("URL", &url), ("API_KEY", "k"), ("AUTH", "header:X-Key"), ("MODEL", "gte-rerank-v2")]).scores("q", &["d1"]).await.unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!(path, "/services/rerank/text-rerank/text-rerank");
        assert_eq!(headers["x-key"], "k");
        assert_eq!(body["input"], json!({"query": "q", "documents": ["d1"]}));
    }

    #[test]
    fn replies_are_read_leniently() {
        let r = remote(&[("PROTOCOL", "template"), ("URL", "http://x"), ("TEMPLATE", "{}"), ("TEMPLATE_RESULTS", "scores")]);
        // Plain scores in document order.
        assert_eq!(r.parse(&json!({"scores": [0.5, 0.25, 0.75]}), 3).unwrap().0, vec![0.5, 0.25, 0.75]);
        // A document left out ranks below the rest.
        let c = remote(&[("PROTOCOL", "cohere"), ("MODEL", "m")]);
        let (scores, units) = c.parse(&json!({"results": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.2}], "meta": {"billed_units": {"search_units": 1}}}), 3).unwrap();
        assert_eq!((scores, units), (vec![0.2, 0.9, 0.2 - 1.0], 1));
        assert!(c.parse(&json!({"results": [{"index": 5, "relevance_score": 1.0}]}), 2).is_err());
        assert!(c.parse(&json!({"error": "x"}), 2).is_err());
    }

    #[test]
    fn settings_are_checked() {
        let get = |vars: &'static [(&'static str, &'static str)]| move |k: &str| vars.iter().find(|(n, _)| *n == k).map(|(_, v)| v.to_string());
        assert!(Remote::from_lookup(get(&[])).unwrap().is_none());
        assert!(Remote::from_lookup(get(&[("TIDEMARK_RERANK_PROTOCOL", "nope")])).is_err());
        assert!(Remote::from_lookup(get(&[("TIDEMARK_RERANK_PROTOCOL", "cohere")])).is_err(), "model required");
        assert!(Remote::from_lookup(get(&[("TIDEMARK_RERANK_PROTOCOL", "tei")])).is_err(), "url required");
        assert!(Remote::from_lookup(get(&[("TIDEMARK_RERANK_PROTOCOL", "template"), ("TIDEMARK_RERANK_URL", "http://x")])).is_err(), "template required");
    }
}
