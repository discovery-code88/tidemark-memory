//! Text embeddings: an OpenAI-compatible `/embeddings` endpoint (for example
//! `text-embedding-v4` on DashScope's compatible mode), or a hashed
//! bag-of-words placeholder for offline development (never for evaluation).

use sha2::{Digest, Sha256};
use std::time::Duration;

#[derive(Clone)]
pub enum Embedder {
    Remote { url: String, key: String, model: String, dims: Option<usize>, batch: usize, parallel: usize, client: reqwest::Client },
    Hash { dims: usize },
}

impl Embedder {
    /// `TIDEMARK_EMBED_URL` (base URL ending in /v1 or the full /embeddings URL),
    /// `TIDEMARK_EMBED_API_KEY`, `TIDEMARK_EMBED_MODEL` (default text-embedding-v4),
    /// `TIDEMARK_EMBED_DIMS` (optional), `TIDEMARK_EMBED_BATCH` (default 10),
    /// `TIDEMARK_EMBED_PARALLEL` (batches of one request sent at once, default 4).
    /// Without a key the hashed placeholder is used.
    pub fn from_env() -> Self {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        match (var("TIDEMARK_EMBED_URL"), var("TIDEMARK_EMBED_API_KEY")) {
            (Some(url), Some(key)) => {
                let url = if url.ends_with("/embeddings") { url } else { format!("{}/embeddings", url.trim_end_matches('/')) };
                Embedder::Remote {
                    url,
                    key,
                    model: var("TIDEMARK_EMBED_MODEL").unwrap_or_else(|| "text-embedding-v4".into()),
                    dims: var("TIDEMARK_EMBED_DIMS").and_then(|v| v.parse().ok()),
                    batch: var("TIDEMARK_EMBED_BATCH").and_then(|v| v.parse().ok()).unwrap_or(10).max(1),
                    parallel: var("TIDEMARK_EMBED_PARALLEL").and_then(|v| v.parse().ok()).unwrap_or(4).max(1),
                    client: reqwest::Client::builder().timeout(Duration::from_secs(120)).build().expect("http client"),
                }
            }
            _ => Embedder::Hash { dims: 512 },
        }
    }

    pub fn label(&self) -> String {
        match self {
            Embedder::Remote { model, dims, .. } => format!("{model}{}", dims.map(|d| format!("@{d}")).unwrap_or_default()),
            Embedder::Hash { dims } => format!("hash-placeholder@{dims}"),
        }
    }

    pub async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, String> {
        self.embed_counted(texts).await.map(|(v, _)| v)
    }

    /// Vectors plus the tokens the provider reported (`usage.total_tokens`;
    /// 0 for the placeholder). Every call adds to `TOKENS_USED`.
    pub async fn embed_counted(&self, texts: &[String]) -> Result<(Vec<Vec<f32>>, u64), String> {
        match self {
            Embedder::Hash { dims } => Ok((texts.iter().map(|t| normalized(hash_vector(t, *dims))).collect(), 0)),
            Embedder::Remote { url, key, model, dims, batch, parallel, client } => {
                // Inputs beyond the model's limit (8192 tokens for text-embedding-v4)
                // are rejected; keep the first characters (about one token each in
                // the worst case, e.g. Chinese). BM25 still sees the whole record.
                let capped = texts.iter().map(|t| if t.chars().count() > 6000 { t.chars().take(6000).collect() } else { t.clone() }).collect::<Vec<String>>();
                // The batches of one request go out `parallel` at a time; results
                // keep the input order.
                let limit = std::sync::Arc::new(tokio::sync::Semaphore::new(*parallel));
                let mut tasks = tokio::task::JoinSet::new();
                for (n, chunk) in capped.chunks(*batch).enumerate() {
                    let mut body = serde_json::json!({"model": model, "input": chunk, "encoding_format": "float"});
                    if let Some(d) = dims {
                        body["dimensions"] = serde_json::json!(d);
                    }
                    let (client, url, key, limit, expected) = (client.clone(), url.clone(), key.clone(), limit.clone(), chunk.len());
                    tasks.spawn(async move {
                        let _permit = limit.acquire_owned().await.map_err(|e| e.to_string())?;
                        embed_batch(&client, &url, &key, &body, expected).await.map(|v| (n, v))
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
    }
}

/// Embedding tokens reported by the provider since the service started.
pub static TOKENS_USED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One batch, retried on 429, server errors and connection errors; returns
/// the vectors and the reported tokens.
async fn embed_batch(client: &reqwest::Client, url: &str, key: &str, body: &serde_json::Value, expected: usize) -> Result<(Vec<Vec<f32>>, u64), String> {
    let mut attempt = 0;
    let value: serde_json::Value = loop {
        attempt += 1;
        let result = client.post(url).bearer_auth(key).json(body).send().await;
        match result {
            Ok(resp) if resp.status().is_success() => break resp.json().await.map_err(|e| format!("embedding reply: {e}"))?,
            Ok(resp) if attempt < 6 && (resp.status().as_u16() == 429 || resp.status().is_server_error()) => {}
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("embedding HTTP {status}: {}", text.chars().take(300).collect::<String>()));
            }
            Err(e) if attempt < 6 => eprintln!("embedding request error (retrying): {e}"),
            Err(e) => return Err(format!("embedding request: {e}")),
        }
        tokio::time::sleep(Duration::from_millis(500 * (1 << attempt.min(6)))).await;
    };
    let mut data = value["data"].as_array().cloned().ok_or("embedding reply without data")?;
    data.sort_by_key(|item| item["index"].as_u64().unwrap_or(0));
    if data.len() != expected {
        return Err(format!("embedding reply has {} vectors for {} inputs", data.len(), expected));
    }
    let tokens = value["usage"]["total_tokens"].as_u64().unwrap_or(0);
    let vectors = data
        .into_iter()
        .map(|item| {
            let v = item["embedding"].as_array().ok_or("embedding without vector")?.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect();
            Ok(normalized(v))
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok((vectors, tokens))
}

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
