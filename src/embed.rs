//! Text embeddings from any provider: native protocols for the common APIs
//! (OpenAI-compatible, Azure OpenAI, Gemini, Cohere, Voyage, Jina, Ollama,
//! Hugging Face TEI), a JSON template for anything else, and a hashed
//! bag-of-words placeholder for offline development (never for real use).
//!
//! Settings (all `TIDEMARK_EMBED_…`):
//! - `PROTOCOL`: openai | azure | gemini | cohere | voyage | jina | ollama | tei | template | hash.
//!   Default: openai when `URL` and `API_KEY` are set (as in 1.0), otherwise the placeholder.
//! - `URL` (hosted providers have a default), `API_KEY`, `MODEL`, `DIMS` (optional output size).
//! - `QUERY_TYPE` / `DOCUMENT_TYPE`: the provider's input types for questions and stored
//!   texts (defaults per protocol; `none` disables). `QUERY_PREFIX` / `DOCUMENT_PREFIX`:
//!   text put before each input (e.g. "query: " / "passage: " for E5 models).
//! - `HEADERS`: extra headers, `Name: value` separated by `;` or newlines.
//!   `AUTH`: bearer | none | header:Name (default per protocol).
//! - `API_VERSION` (Azure), `TEMPLATE`, `TEMPLATE_VECTORS`, `TEMPLATE_TOKENS` (template).
//! - `MAX_CHARS` (input cap, default 6000), `BATCH` (texts per request), `PARALLEL`
//!   (requests of one call at a time, default 4).

use crate::http::{self, Auth};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Whether a text is a question (search) or a stored text (Add). Providers that
/// distinguish the two give better matches when told which is which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Query,
    Document,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    OpenAi,
    Azure,
    Gemini,
    Cohere,
    Voyage,
    Jina,
    Ollama,
    Tei,
    Template,
}

impl Protocol {
    fn parse(name: &str) -> Option<Protocol> {
        Some(match name.trim().to_ascii_lowercase().as_str() {
            "openai" | "openai-compatible" | "openai_compatible" => Protocol::OpenAi,
            "azure" | "azure-openai" => Protocol::Azure,
            "gemini" | "google" => Protocol::Gemini,
            "cohere" => Protocol::Cohere,
            "voyage" | "voyageai" => Protocol::Voyage,
            "jina" => Protocol::Jina,
            "ollama" => Protocol::Ollama,
            "tei" | "huggingface-tei" => Protocol::Tei,
            "template" | "custom" => Protocol::Template,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Protocol::OpenAi => "openai",
            Protocol::Azure => "azure",
            Protocol::Gemini => "gemini",
            Protocol::Cohere => "cohere",
            Protocol::Voyage => "voyage",
            Protocol::Jina => "jina",
            Protocol::Ollama => "ollama",
            Protocol::Tei => "tei",
            Protocol::Template => "template",
        }
    }

    fn default_url(self) -> Option<&'static str> {
        match self {
            Protocol::OpenAi => Some("https://api.openai.com/v1"),
            Protocol::Gemini => Some("https://generativelanguage.googleapis.com/v1beta"),
            Protocol::Cohere => Some("https://api.cohere.com/v2"),
            Protocol::Voyage => Some("https://api.voyageai.com/v1"),
            Protocol::Jina => Some("https://api.jina.ai/v1"),
            Protocol::Ollama => Some("http://127.0.0.1:11434"),
            Protocol::Azure | Protocol::Tei | Protocol::Template => None,
        }
    }

    /// (query type, document type) the provider expects by default.
    fn default_types(self) -> (Option<&'static str>, Option<&'static str>) {
        match self {
            Protocol::Cohere => (Some("search_query"), Some("search_document")),
            Protocol::Voyage => (Some("query"), Some("document")),
            Protocol::Jina => (Some("retrieval.query"), Some("retrieval.passage")),
            Protocol::Gemini => (Some("RETRIEVAL_QUERY"), Some("RETRIEVAL_DOCUMENT")),
            _ => (None, None),
        }
    }

    fn default_batch(self) -> usize {
        match self {
            // DashScope's compatible mode takes at most 10 inputs per request.
            Protocol::OpenAi | Protocol::Azure | Protocol::Template => 10,
            Protocol::Gemini => 100,
            Protocol::Cohere => 96,
            Protocol::Voyage => 128,
            Protocol::Jina => 64,
            Protocol::Ollama => 16,
            Protocol::Tei => 32,
        }
    }

    fn default_auth(self) -> Auth {
        match self {
            Protocol::Azure => Auth::Header("api-key".into()),
            Protocol::Gemini => Auth::Header("x-goog-api-key".into()),
            _ => Auth::Bearer,
        }
    }
}

#[derive(Clone)]
pub struct Remote {
    protocol: Protocol,
    url: String,
    key: Option<String>,
    model: String,
    dims: Option<usize>,
    batch: usize,
    parallel: usize,
    max_chars: usize,
    query_type: Option<String>,
    document_type: Option<String>,
    query_prefix: String,
    document_prefix: String,
    headers: Vec<(String, String)>,
    auth: Auth,
    api_version: String,
    template: Option<Value>,
    template_vectors: String,
    template_tokens: Option<String>,
    client: reqwest::Client,
}

#[derive(Clone)]
pub enum Embedder {
    Remote(Box<Remote>),
    /// A sentence-embedding model run on this machine by ONNX Runtime (`TIDEMARK_EMBED_PROTOCOL=local`).
    Local(std::sync::Arc<Local>),
    Hash { dims: usize },
}

/// A local embedding model: `TIDEMARK_EMBED_DIR` names a directory with `model.onnx` and `tokenizer.json` (e.g.
/// intfloat/multilingual-e5-small or BAAI/bge-small exported to ONNX). No request leaves the machine and nothing
/// is billed. The vector of a text is the mean of its token states (or the model's own sentence output when it has
/// one), normalized. Models that expect a marker in front of the text get it from `TIDEMARK_EMBED_QUERY_PREFIX`
/// and `TIDEMARK_EMBED_DOCUMENT_PREFIX`; a directory whose name contains "e5" defaults to "query: " and
/// "passage: ", as those models were trained. `TIDEMARK_EMBED_MAX_TOKENS` (default 512) is the model's input
/// limit: text beyond it does not reach the model (BM25 still sees the whole record). `TIDEMARK_EMBED_THREADS`.
pub struct Local {
    session: std::sync::Mutex<ort::session::Session>,
    tokenizer: tokenizers::Tokenizer,
    input_names: Vec<String>,
    name: String,
    dims: usize,
    query_prefix: String,
    document_prefix: String,
}

impl Local {
    pub fn load(dir: &std::path::Path, threads: usize, max_tokens: usize, query_prefix: Option<String>, document_prefix: Option<String>) -> Result<Self, String> {
        let mut tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| format!("{}: {e}", dir.display()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams { max_length: max_tokens, ..tokenizers::TruncationParams::default() }))
            .map_err(|e| e.to_string())?;
        tokenizer.with_padding(None);
        let session = crate::rerank::load_session(&dir.join("model.onnx"), threads)?;
        let input_names = session.inputs().iter().map(|input| input.name().to_string()).collect();
        let name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let e5 = name.to_lowercase().contains("e5");
        let mut local = Local {
            session: std::sync::Mutex::new(session),
            tokenizer,
            input_names,
            name,
            dims: 0,
            query_prefix: query_prefix.unwrap_or_else(|| if e5 { "query: ".into() } else { String::new() }),
            document_prefix: document_prefix.unwrap_or_else(|| if e5 { "passage: ".into() } else { String::new() }),
        };
        local.dims = local.vectors(&["dimension probe".to_string()], Kind::Document)?.first().map_or(0, Vec::len);
        if local.dims == 0 {
            return Err(format!("{}: the model returned no vector", dir.display()));
        }
        Ok(local)
    }

    /// The model's vector length, probed at load.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Normalized vectors, in input order.
    pub fn vectors(&self, texts: &[String], kind: Kind) -> Result<Vec<Vec<f32>>, String> {
        let prefix = if kind == Kind::Query { &self.query_prefix } else { &self.document_prefix };
        let pad_id = self.tokenizer.token_to_id("<pad>").or_else(|| self.tokenizer.token_to_id("[PAD]")).map(i64::from).unwrap_or(0);
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(16) {
            let encodings = self.tokenizer.encode_batch(chunk.iter().map(|t| format!("{prefix}{t}")).collect::<Vec<_>>(), true).map_err(|e| e.to_string())?;
            let rows = encodings.len();
            let width = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(1).max(1);
            let (mut ids, mut mask, types) = (vec![pad_id; rows * width], vec![0_i64; rows * width], vec![0_i64; rows * width]);
            for (row, encoding) in encodings.iter().enumerate() {
                for (column, (id, attention)) in encoding.get_ids().iter().zip(encoding.get_attention_mask()).enumerate() {
                    ids[row * width + column] = i64::from(*id);
                    mask[row * width + column] = i64::from(*attention);
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
                inputs.push((name.clone(), ort::value::Tensor::from_array((shape, data)).map_err(|e| e.to_string())?.into_dyn()));
            }
            let mut session = self.session.lock().map_err(|_| "embedding session poisoned".to_string())?;
            let outputs = session.run(inputs).map_err(|e| e.to_string())?;
            let (out_shape, values) = outputs[0].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
            let dims = out_shape.last().copied().unwrap_or(0).max(0) as usize;
            if dims == 0 {
                return Err("the model returned an empty output".into());
            }
            if values.len() == rows * dims {
                // The model's own sentence vector.
                out.extend(values.chunks(dims).map(|v| normalized(v.to_vec())));
            } else if values.len() == rows * width * dims {
                // Token states: the mean over the text's own tokens.
                for row in 0..rows {
                    let mut sum = vec![0.0f32; dims];
                    let mut count = 0.0f32;
                    for column in 0..width {
                        if mask[row * width + column] == 1 {
                            count += 1.0;
                            let at = (row * width + column) * dims;
                            for (s, v) in sum.iter_mut().zip(&values[at..at + dims]) {
                                *s += v;
                            }
                        }
                    }
                    out.push(normalized(sum.into_iter().map(|s| s / count.max(1.0)).collect()));
                }
            } else {
                return Err(format!("unexpected output size {} for {rows} texts of {width} tokens", values.len()));
            }
        }
        Ok(out)
    }
}

impl Embedder {
    pub fn from_env() -> Self {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    /// The embedder described by `TIDEMARK_EMBED_*` values from `get`.
    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Self {
        let var = |k: &str| http::setting(&get, &format!("TIDEMARK_EMBED_{k}"));
        let url = var("URL");
        let protocol = match var("PROTOCOL") {
            Some(p) if p.eq_ignore_ascii_case("hash") || p.eq_ignore_ascii_case("placeholder") => return Embedder::Hash { dims: 512 },
            Some(p) if p.eq_ignore_ascii_case("local") || p.eq_ignore_ascii_case("onnx") => {
                let dir = var("DIR").unwrap_or_else(|| panic!("TIDEMARK_EMBED_DIR is required for the local protocol (a directory with model.onnx and tokenizer.json)"));
                let threads = var("THREADS").and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));
                let max_tokens = var("MAX_TOKENS").and_then(|v| v.parse().ok()).unwrap_or(512).max(16);
                let local = Local::load(std::path::Path::new(&dir), threads, max_tokens, get("TIDEMARK_EMBED_QUERY_PREFIX"), get("TIDEMARK_EMBED_DOCUMENT_PREFIX"))
                    .unwrap_or_else(|e| panic!("local embedding model: {e}"));
                return Embedder::Local(std::sync::Arc::new(local));
            }
            Some(p) => Protocol::parse(&p).unwrap_or_else(|| panic!("unknown TIDEMARK_EMBED_PROTOCOL {p:?}")),
            // As in 1.0: without a named protocol, a URL needs a key (a keyless
            // local server names its protocol, e.g. ollama, tei or openai).
            None if url.is_some() && var("API_KEY").is_some() => Protocol::OpenAi,
            None => return Embedder::Hash { dims: 512 },
        };
        let url = url
            .or_else(|| protocol.default_url().map(String::from))
            .unwrap_or_else(|| panic!("TIDEMARK_EMBED_URL is required for the {} protocol", protocol.name()));
        let (qt, dt) = protocol.default_types();
        let input_type = |k: &str, default: Option<&str>| match var(k) {
            Some(v) if v.eq_ignore_ascii_case("none") => None,
            Some(v) => Some(v),
            None => default.map(String::from),
        };
        let auth = Auth::parse(var("AUTH").as_deref(), "TIDEMARK_EMBED_AUTH").unwrap_or_else(|| protocol.default_auth());
        let template = var("TEMPLATE").map(|t| serde_json::from_str::<Value>(&t).unwrap_or_else(|e| panic!("TIDEMARK_EMBED_TEMPLATE is not JSON: {e}")));
        if protocol == Protocol::Template && template.is_none() {
            panic!("the template protocol needs TIDEMARK_EMBED_TEMPLATE (a JSON body containing \"{{{{inputs}}}}\")");
        }
        let model = var("MODEL").unwrap_or_else(|| match protocol {
            Protocol::OpenAi => "text-embedding-v4".into(),
            Protocol::Tei | Protocol::Template => String::new(),
            p => panic!("TIDEMARK_EMBED_MODEL is required for the {} protocol", p.name()),
        });
        Embedder::Remote(Box::new(Remote {
            protocol,
            url: url.trim_end_matches('/').to_string(),
            key: var("API_KEY"),
            model,
            dims: var("DIMS").and_then(|v| v.parse().ok()),
            batch: var("BATCH").and_then(|v| v.parse().ok()).unwrap_or(protocol.default_batch()).max(1),
            parallel: var("PARALLEL").and_then(|v| v.parse().ok()).unwrap_or(4).max(1),
            max_chars: var("MAX_CHARS").and_then(|v| v.parse().ok()).unwrap_or(6000).max(1),
            query_type: input_type("QUERY_TYPE", qt),
            document_type: input_type("DOCUMENT_TYPE", dt),
            query_prefix: get("TIDEMARK_EMBED_QUERY_PREFIX").unwrap_or_default(),
            document_prefix: get("TIDEMARK_EMBED_DOCUMENT_PREFIX").unwrap_or_default(),
            headers: var("HEADERS").map(|h| http::parse_headers(&h)).unwrap_or_default(),
            auth,
            api_version: var("API_VERSION").unwrap_or_else(|| "2024-10-21".into()),
            template,
            template_vectors: var("TEMPLATE_VECTORS").unwrap_or_else(|| "data.*.embedding".into()),
            template_tokens: var("TEMPLATE_TOKENS"),
            client: http::client(),
        }))
    }

    /// Identity of the stored vectors (checked against the database): the model,
    /// its output size, and what else changes stored vectors (document type and
    /// prefix). An OpenAI-compatible setup without those keeps version 1.0's label.
    pub fn label(&self) -> String {
        match self {
            Embedder::Hash { dims } => format!("hash-placeholder@{dims}"),
            Embedder::Local(l) => {
                let mut label = format!("local:{}@{}", l.name, l.dims);
                if !l.document_prefix.is_empty() {
                    label.push_str(&format!("|docprefix:{}", l.document_prefix));
                }
                label
            }
            Embedder::Remote(r) => {
                let mut label = if r.protocol == Protocol::OpenAi { r.model.clone() } else { format!("{}:{}", r.protocol.name(), r.model) };
                if let Some(d) = r.dims {
                    label.push_str(&format!("@{d}"));
                }
                if let Some(t) = &r.document_type {
                    label.push_str(&format!("|doc:{t}"));
                }
                if !r.document_prefix.is_empty() {
                    label.push_str(&format!("|docprefix:{}", r.document_prefix));
                }
                label
            }
        }
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, Embedder::Remote(_))
    }

    pub async fn embed(&self, texts: &[String], kind: Kind) -> Result<Vec<Vec<f32>>, String> {
        self.embed_counted(texts, kind).await.map(|(v, _)| v)
    }

    /// Vectors (normalized) plus the tokens the provider reported (0 when it
    /// reports none, and for the placeholder). Every call adds to `TOKENS_USED`.
    pub async fn embed_counted(&self, texts: &[String], kind: Kind) -> Result<(Vec<Vec<f32>>, u64), String> {
        let r = match self {
            Embedder::Hash { dims } => return Ok((texts.iter().map(|t| normalized(hash_vector(t, *dims))).collect(), 0)),
            // CPU work on this machine: nothing is sent and nothing is billed.
            Embedder::Local(l) => return tokio::task::block_in_place(|| l.vectors(texts, kind)).map(|v| (v, 0)),
            Embedder::Remote(r) => r,
        };
        let prefix = if kind == Kind::Query { &r.query_prefix } else { &r.document_prefix };
        // Inputs beyond a model's limit are rejected; keep the first characters
        // (about one token each in the worst case, e.g. Chinese). BM25 still
        // sees the whole record.
        let inputs = texts
            .iter()
            .map(|t| {
                let t = if t.chars().count() > r.max_chars { t.chars().take(r.max_chars).collect::<String>() } else { t.clone() };
                format!("{prefix}{t}")
            })
            .collect::<Vec<String>>();
        // The batches of one call go out `parallel` at a time; results keep the input order.
        let limit = std::sync::Arc::new(tokio::sync::Semaphore::new(r.parallel));
        let mut tasks = tokio::task::JoinSet::new();
        for (n, chunk) in inputs.chunks(r.batch).enumerate() {
            let (url, headers, body) = r.request(chunk, kind);
            let (limit, expected, remote) = (limit.clone(), chunk.len(), r.clone());
            tasks.spawn(async move {
                let _permit = limit.acquire_owned().await.map_err(|e| e.to_string())?;
                let value = http::post(&remote.client, &url, &headers, &body, "embedding").await?;
                remote.parse(&value, expected).map(|v| (n, v))
            });
        }
        let mut parts = Vec::new();
        while let Some(joined) = tasks.join_next().await {
            parts.push(joined.map_err(|e| format!("embedding task: {e}"))??);
        }
        parts.sort_by_key(|(n, _)| *n);
        let tokens = parts.iter().map(|(_, (_, t))| *t).sum::<u64>();
        TOKENS_USED.fetch_add(tokens, std::sync::atomic::Ordering::Relaxed);
        Ok((parts.into_iter().flat_map(|(_, (v, _))| v).collect(), tokens))
    }
}

impl Remote {
    fn input_type(&self, kind: Kind) -> Option<&str> {
        if kind == Kind::Query { self.query_type.as_deref() } else { self.document_type.as_deref() }
    }

    /// URL, headers and JSON body of one batch.
    fn request(&self, inputs: &[String], kind: Kind) -> (String, Vec<(String, String)>, Value) {
        let ty = self.input_type(kind);
        let with = |mut b: Value, key: &str, v: Option<Value>| {
            if let Some(v) = v {
                b[key] = v;
            }
            b
        };
        let dims = self.dims.map(|d| json!(d));
        let ty_json = ty.map(|t| json!(t));
        let (url, body) = match self.protocol {
            Protocol::OpenAi => {
                let url = if self.url.ends_with("/embeddings") { self.url.clone() } else { format!("{}/embeddings", self.url) };
                (url, with(json!({"model": self.model, "input": inputs, "encoding_format": "float"}), "dimensions", dims))
            }
            Protocol::Azure => {
                let base = if self.url.contains("/deployments/") { self.url.clone() } else { format!("{}/openai/deployments/{}/embeddings", self.url, self.model) };
                let url = if base.contains("api-version=") { base } else { format!("{base}{}api-version={}", if base.contains('?') { "&" } else { "?" }, self.api_version) };
                (url, with(json!({"input": inputs}), "dimensions", dims))
            }
            Protocol::Gemini => {
                let model = if self.model.starts_with("models/") { self.model.clone() } else { format!("models/{}", self.model) };
                let requests = inputs
                    .iter()
                    .map(|t| with(with(json!({"model": model, "content": {"parts": [{"text": t}]}}), "taskType", ty_json.clone()), "outputDimensionality", dims.clone()))
                    .collect::<Vec<_>>();
                (format!("{}/{model}:batchEmbedContents", self.url), json!({"requests": requests}))
            }
            Protocol::Cohere => {
                let b = with(json!({"model": self.model, "texts": inputs, "embedding_types": ["float"]}), "input_type", ty_json);
                (format!("{}/embed", self.url), with(b, "output_dimension", dims))
            }
            Protocol::Voyage => {
                let b = with(json!({"model": self.model, "input": inputs}), "input_type", ty_json);
                (format!("{}/embeddings", self.url), with(b, "output_dimension", dims))
            }
            Protocol::Jina => {
                let b = with(json!({"model": self.model, "input": inputs}), "task", ty_json);
                (format!("{}/embeddings", self.url), with(b, "dimensions", dims))
            }
            Protocol::Ollama => (format!("{}/api/embed", self.url), with(json!({"model": self.model, "input": inputs}), "dimensions", dims)),
            Protocol::Tei => (format!("{}/embed", self.url), json!({"inputs": inputs, "truncate": true})),
            Protocol::Template => {
                let values = [("inputs", json!(inputs)), ("model", json!(self.model)), ("input_type", json!(ty.unwrap_or(""))), ("dims", self.dims.map_or(Value::Null, |d| json!(d)))];
                (self.url.clone(), http::fill(self.template.as_ref().expect("template"), &values))
            }
        };
        (url, http::headers(self.key.as_deref(), &self.auth, &self.headers), body)
    }

    /// The batch's vectors (in input order, normalized) and reported tokens.
    fn parse(&self, value: &Value, expected: usize) -> Result<(Vec<Vec<f32>>, u64), String> {
        fn as_vector(v: &Value) -> Result<Vec<f32>, String> {
            Ok(v.as_array().ok_or("embedding is not an array")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
        }
        fn all(list: &[Value], f: impl Fn(&Value) -> &Value) -> Result<Vec<Vec<f32>>, String> {
            list.iter().map(|v| as_vector(f(v))).collect()
        }
        let (vectors, tokens) = match self.protocol {
            Protocol::OpenAi | Protocol::Azure | Protocol::Voyage | Protocol::Jina => {
                let mut data = value["data"].as_array().cloned().ok_or("embedding reply without data")?;
                data.sort_by_key(|item| item["index"].as_u64().unwrap_or(0));
                (all(&data, |d| &d["embedding"])?, value["usage"]["total_tokens"].as_u64().unwrap_or(0))
            }
            Protocol::Gemini => (all(value["embeddings"].as_array().ok_or("embedding reply without embeddings")?, |e| &e["values"])?, 0),
            Protocol::Cohere => {
                // v2 nests by type ("float"); v1 returns the list directly.
                let list = value["embeddings"]["float"].as_array().or_else(|| value["embeddings"].as_array()).ok_or("embedding reply without embeddings")?;
                (all(list, |v| v)?, value["meta"]["billed_units"]["input_tokens"].as_u64().unwrap_or(0))
            }
            Protocol::Ollama => {
                (all(value["embeddings"].as_array().ok_or("embedding reply without embeddings")?, |v| v)?, value["prompt_eval_count"].as_u64().unwrap_or(0))
            }
            Protocol::Tei => (all(value.as_array().ok_or("embedding reply is not a list")?, |v| v)?, 0),
            Protocol::Template => {
                let found = http::select(value, &self.template_vectors).into_iter().cloned().collect::<Vec<_>>();
                let tokens = self.template_tokens.as_ref().and_then(|p| http::select(value, p).first().and_then(|v| v.as_u64())).unwrap_or(0);
                (all(&found, |v| v)?, tokens)
            }
        };
        if vectors.len() != expected {
            return Err(format!("embedding reply has {} vectors for {} inputs", vectors.len(), expected));
        }
        if vectors.iter().any(|v| v.is_empty()) {
            return Err("embedding reply contains an empty vector".into());
        }
        Ok((vectors.into_iter().map(normalized).collect(), tokens))
    }
}

/// Embedding tokens reported by the provider since the service started.
pub static TOKENS_USED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn hash_vector(text: &str, dims: usize) -> Vec<f32> {
    let mut v = vec![0f32; dims];
    for token in crate::search::tokens(text) {
        let h = Sha256::digest(token.as_bytes());
        let i = u32::from_le_bytes([h[0], h[1], h[2], h[3]]) as usize % dims;
        v[i] += if h[4] & 1 == 0 { 1.0 } else { -1.0 };
    }
    v
}

pub fn normalized(mut v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 {
        v.iter_mut().for_each(|x| *x /= n);
    }
    v
}

/// Cosine of two normalized vectors; vectors of different lengths (from
/// different models) are not comparable and score 0. Startup refuses such a
/// database, so this only guards against a mixed one.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    type Seen = Arc<Mutex<Vec<(String, HashMap<String, String>, Value)>>>;

    /// A local server answering every POST like `protocol`'s provider would:
    /// vector i of a batch is [1, i+1]. Records path+query, headers and body.
    async fn mock(protocol: Protocol) -> (String, Seen) {
        let seen: Seen = Arc::default();
        let log = seen.clone();
        let app = axum::Router::new().fallback(move |uri: axum::http::Uri, headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
            let log = log.clone();
            async move {
                let n = ["input", "texts", "inputs", "requests", "items"].iter().find_map(|k| body[k].as_array().map(Vec::len)).unwrap_or(1);
                let vecs = (0..n).map(|i| json!([1.0, (i + 1) as f64])).collect::<Vec<_>>();
                let reply = match protocol {
                    // Out of order on purpose: the index decides.
                    Protocol::OpenAi | Protocol::Azure | Protocol::Voyage | Protocol::Jina => json!({
                        "data": vecs.iter().enumerate().rev().map(|(i, v)| json!({"index": i, "embedding": v})).collect::<Vec<_>>(),
                        "usage": {"total_tokens": 7}
                    }),
                    Protocol::Gemini => json!({"embeddings": vecs.iter().map(|v| json!({"values": v})).collect::<Vec<_>>()}),
                    Protocol::Cohere => json!({"embeddings": {"float": vecs}, "meta": {"billed_units": {"input_tokens": 7}}}),
                    Protocol::Ollama => json!({"embeddings": vecs, "prompt_eval_count": 7}),
                    Protocol::Tei => json!(vecs),
                    Protocol::Template => json!({"result": {"vectors": vecs, "used": 7}}),
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

    fn embedder(vars: &[(&str, &str)]) -> Embedder {
        let map = vars.iter().map(|(k, v)| (format!("TIDEMARK_EMBED_{k}"), v.to_string())).collect::<HashMap<_, _>>();
        Embedder::from_lookup(|k| map.get(k).cloned())
    }

    fn texts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("text {i}")).collect()
    }

    #[test]
    fn version_1_settings_keep_their_label() {
        let e = embedder(&[("URL", "https://example.test/v1"), ("API_KEY", "k"), ("MODEL", "text-embedding-v4"), ("DIMS", "1024")]);
        assert_eq!(e.label(), "text-embedding-v4@1024");
        assert_eq!(embedder(&[]).label(), "hash-placeholder@512");
        assert_eq!(embedder(&[("PROTOCOL", "cohere"), ("MODEL", "embed-v4.0")]).label(), "cohere:embed-v4.0|doc:search_document");
        assert_eq!(embedder(&[("URL", "http://x"), ("API_KEY", "k"), ("DOCUMENT_PREFIX", "passage: ")]).label(), "text-embedding-v4|docprefix:passage: ");
        // 1.0: a URL without a key is the placeholder, unless a protocol is named.
        assert_eq!(embedder(&[("URL", "http://x")]).label(), "hash-placeholder@512");
        assert_eq!(embedder(&[("PROTOCOL", "openai"), ("URL", "http://x"), ("MODEL", "local")]).label(), "local");
    }

    #[tokio::test]
    async fn every_protocol_round_trips_in_order() {
        for protocol in [Protocol::OpenAi, Protocol::Azure, Protocol::Gemini, Protocol::Cohere, Protocol::Voyage, Protocol::Jina, Protocol::Ollama, Protocol::Tei, Protocol::Template] {
            let (url, seen) = mock(protocol).await;
            let template = r#"{"items": "{{inputs}}", "model": "{{model}}", "kind": "{{input_type}}"}"#;
            let e = embedder(&[
                ("PROTOCOL", protocol.name()),
                ("URL", &url),
                ("API_KEY", "secret"),
                ("MODEL", "m1"),
                ("BATCH", "2"),
                ("TEMPLATE", template),
                ("TEMPLATE_VECTORS", "result.vectors"),
                ("TEMPLATE_TOKENS", "result.used"),
            ]);
            let (vectors, _) = e.embed_counted(&texts(5), Kind::Document).await.unwrap_or_else(|err| panic!("{protocol:?}: {err}"));
            // Batches of 2: within each batch the second vector is [1, 2].
            let expected = [1.0f32, 2.0, 1.0, 2.0, 1.0].iter().map(|&y| normalized(vec![1.0, y])).collect::<Vec<_>>();
            assert_eq!(vectors, expected, "{protocol:?}");
            assert_eq!(seen.lock().unwrap().len(), 3, "{protocol:?}");
        }
    }

    #[tokio::test]
    async fn requests_follow_each_providers_format() {
        let first = |seen: &Seen| seen.lock().unwrap()[0].clone();

        let (url, seen) = mock(Protocol::OpenAi).await;
        let (_, tokens) = embedder(&[("URL", &format!("{url}/v1")), ("API_KEY", "secret"), ("DIMS", "2")]).embed_counted(&texts(1), Kind::Query).await.unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!(path, "/v1/embeddings");
        assert_eq!(headers["authorization"], "Bearer secret");
        assert_eq!(body, json!({"model": "text-embedding-v4", "input": ["text 0"], "encoding_format": "float", "dimensions": 2}));
        assert_eq!(tokens, 7);

        let (url, seen) = mock(Protocol::Azure).await;
        embedder(&[("PROTOCOL", "azure"), ("URL", &url), ("API_KEY", "secret"), ("MODEL", "dep1")]).embed(&texts(1), Kind::Query).await.unwrap();
        let (path, headers, _) = first(&seen);
        assert_eq!(path, "/openai/deployments/dep1/embeddings?api-version=2024-10-21");
        assert_eq!(headers["api-key"], "secret");
        assert!(!headers.contains_key("authorization"));

        let (url, seen) = mock(Protocol::Gemini).await;
        embedder(&[("PROTOCOL", "gemini"), ("URL", &url), ("API_KEY", "secret"), ("MODEL", "gemini-embedding-001"), ("DIMS", "768")]).embed(&texts(1), Kind::Query).await.unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!(path, "/models/gemini-embedding-001:batchEmbedContents");
        assert_eq!(headers["x-goog-api-key"], "secret");
        assert_eq!(body["requests"][0]["taskType"], "RETRIEVAL_QUERY");
        assert_eq!(body["requests"][0]["outputDimensionality"], 768);
        assert_eq!(body["requests"][0]["content"]["parts"][0]["text"], "text 0");

        let (url, seen) = mock(Protocol::Cohere).await;
        let e = embedder(&[("PROTOCOL", "cohere"), ("URL", &url), ("MODEL", "embed-v4.0")]);
        e.embed(&texts(1), Kind::Query).await.unwrap();
        e.embed(&texts(1), Kind::Document).await.unwrap();
        let log = seen.lock().unwrap().clone();
        assert_eq!(log[0].0, "/embed");
        assert_eq!((log[0].2["input_type"].as_str(), log[1].2["input_type"].as_str()), (Some("search_query"), Some("search_document")));
        assert!(!log[0].1.contains_key("authorization"), "no key, no auth header");

        let (url, seen) = mock(Protocol::Jina).await;
        embedder(&[("PROTOCOL", "jina"), ("URL", &url), ("MODEL", "jina-embeddings-v3"), ("DOCUMENT_TYPE", "none")]).embed(&texts(1), Kind::Document).await.unwrap();
        assert!(first(&seen).2.get("task").is_none(), "DOCUMENT_TYPE=none sends no task");

        let (url, seen) = mock(Protocol::Ollama).await;
        embedder(&[("PROTOCOL", "ollama"), ("URL", &url), ("MODEL", "nomic-embed-text"), ("QUERY_PREFIX", "search_query: ")]).embed(&texts(1), Kind::Query).await.unwrap();
        let (path, _, body) = first(&seen);
        assert_eq!(path, "/api/embed");
        assert_eq!(body["input"], json!(["search_query: text 0"]));

        let (url, seen) = mock(Protocol::Tei).await;
        embedder(&[("PROTOCOL", "tei"), ("URL", &url), ("MAX_CHARS", "3"), ("HEADERS", "X-One: 1; X-Two: two")]).embed(&texts(1), Kind::Document).await.unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!(path, "/embed");
        assert_eq!(body, json!({"inputs": ["tex"], "truncate": true}));
        assert_eq!((headers["x-one"].as_str(), headers["x-two"].as_str()), ("1", "two"));

        let (url, seen) = mock(Protocol::Template).await;
        let (_, tokens) = embedder(&[
            ("PROTOCOL", "template"),
            ("URL", &format!("{url}/custom/path")),
            ("API_KEY", "secret"),
            ("AUTH", "header:X-Key"),
            ("MODEL", "m1"),
            ("QUERY_TYPE", "q"),
            ("DIMS", "64"),
            ("TEMPLATE", r#"{"items": "{{inputs}}", "opts": {"model": "{{model}}", "kind": "{{input_type}}", "size": "{{dims}}"}}"#),
            ("TEMPLATE_VECTORS", "result.vectors"),
            ("TEMPLATE_TOKENS", "result.used"),
        ])
        .embed_counted(&texts(2), Kind::Query)
        .await
        .unwrap();
        let (path, headers, body) = first(&seen);
        assert_eq!(path, "/custom/path");
        assert_eq!(headers["x-key"], "secret");
        assert_eq!(body, json!({"items": ["text 0", "text 1"], "opts": {"model": "m1", "kind": "q", "size": 64}}));
        assert_eq!(tokens, 7);
    }

    #[tokio::test]
    async fn a_reply_with_the_wrong_number_of_vectors_is_an_error() {
        let (url, _) = mock(Protocol::Template).await;
        let e = embedder(&[("PROTOCOL", "template"), ("URL", &url), ("TEMPLATE", r#"{"items": ["only one"]}"#), ("TEMPLATE_VECTORS", "result.vectors")]);
        let err = e.embed(&texts(3), Kind::Document).await.unwrap_err();
        assert!(err.contains("1 vectors for 3 inputs"), "{err}");
    }
}
