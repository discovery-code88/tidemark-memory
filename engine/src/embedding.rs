use crate::provider_egress::{ProviderEgressGateway, ProviderEgressRequest};
use serde::Deserialize;
use std::env;

const DEFAULT_DIM: usize = 64;

#[derive(Debug, Clone)]
pub enum EmbeddingMode {
    Local,
    OpenAICompat {
        base_url: String,
        api_key: String,
        model: String,
    },
}

#[derive(Clone)]
pub struct EmbeddingClient {
    mode: EmbeddingMode,
    egress: ProviderEgressGateway,
    dims: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct EmbeddingScope {
    pub(crate) tenant_id: String,
    pub(crate) workspace_id: String,
    pub(crate) explicit_provider_approval: bool,
}

impl EmbeddingScope {
    pub(crate) fn new(
        tenant_id: impl Into<String>,
        workspace_id: impl Into<String>,
        explicit_provider_approval: bool,
    ) -> Self {
        Self {
            tenant_id: tenant_id.into(),
            workspace_id: workspace_id.into(),
            explicit_provider_approval,
        }
    }
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    pub fn from_env() -> Self {
        let provider = env::var("EMBEDDING_PROVIDER").unwrap_or_else(|_| "local".to_string());
        let requested_dims = env::var("EMBEDDING_DIMENSIONS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(DEFAULT_DIM)
            .max(1);
        let dims = if requested_dims == DEFAULT_DIM {
            DEFAULT_DIM
        } else {
            eprintln!(
                "EMBEDDING_DIMENSIONS={} is not supported by current schema (vector(64)); fallback to 64",
                requested_dims
            );
            DEFAULT_DIM
        };

        let mode = match provider.as_str() {
            "openai_compat" => {
                let base_url = env::var("EMBEDDING_BASE_URL")
                    .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());
                let api_key = env::var("EMBEDDING_API_KEY").unwrap_or_default();
                let model = env::var("EMBEDDING_MODEL")
                    .unwrap_or_else(|_| "text-embedding-3-small".to_string());
                if api_key.is_empty() {
                    EmbeddingMode::Local
                } else {
                    EmbeddingMode::OpenAICompat {
                        base_url,
                        api_key,
                        model,
                    }
                }
            }
            _ => EmbeddingMode::Local,
        };

        Self {
            mode,
            egress: ProviderEgressGateway::from_env(),
            dims,
        }
    }

    pub fn mode_name(&self) -> &'static str {
        match self.mode {
            EmbeddingMode::Local => "local",
            EmbeddingMode::OpenAICompat { .. } => "openai_compat",
        }
    }

    pub fn provider_label(&self) -> &'static str {
        self.mode_name()
    }

    pub fn model_label(&self) -> &str {
        match &self.mode {
            EmbeddingMode::Local => "hash-embed-v1",
            EmbeddingMode::OpenAICompat { model, .. } => model.as_str(),
        }
    }

    

    pub(crate) async fn embed_scoped(
        &self,
        text: &str,
        scope: &EmbeddingScope,
    ) -> Result<Vec<f32>, String> {
        self.embed_with_scope(text, Some(scope)).await
    }

    async fn embed_with_scope(
        &self,
        text: &str,
        scope: Option<&EmbeddingScope>,
    ) -> Result<Vec<f32>, String> {
        match &self.mode {
            EmbeddingMode::Local => Ok(hash_embed(text, self.dims)),
            EmbeddingMode::OpenAICompat {
                base_url,
                api_key,
                model,
            } => {
                let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
                let payload = serde_json::json!({
                    "model": model,
                    "input": text,
                });
                let mut request = ProviderEgressRequest::from_env(
                    "service:embedding",
                    "memory_embedding",
                    "openai_compat",
                    model,
                    "text",
                    &["text", "embedding_input"],
                    payload,
                );
                if let Some(scope) = scope {
                    request = request
                        .with_workspace_scope(&scope.tenant_id, &scope.workspace_id)
                        .with_explicit_user_approval(scope.explicit_provider_approval);
                }
                let resp = self
                    .egress
                    .post_json(&url, Some(api_key), request)
                    .await
                    .map_err(|e| format!("http_error:{e}"))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(format!("provider_error:{status}:{body}"));
                }

                let parsed: EmbeddingResponse =
                    resp.json().await.map_err(|e| format!("parse_error:{e}"))?;
                let raw = parsed
                    .data
                    .first()
                    .map(|d| d.embedding.clone())
                    .ok_or_else(|| "empty_embedding".to_string())?;
                Ok(normalize_dims(raw, self.dims))
            }
        }
    }

    

    pub(crate) async fn embed_many_scoped(
        &self,
        texts: &[String],
        scope: &EmbeddingScope,
    ) -> Result<Vec<Vec<f32>>, String> {
        self.embed_many_with_scope(texts, Some(scope)).await
    }

    async fn embed_many_with_scope(
        &self,
        texts: &[String],
        scope: Option<&EmbeddingScope>,
    ) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        match &self.mode {
            EmbeddingMode::Local => Ok(texts
                .iter()
                .map(|t| hash_embed(t, self.dims))
                .collect::<Vec<_>>()),
            EmbeddingMode::OpenAICompat {
                base_url,
                api_key,
                model,
            } => {
                let url = format!("{}/embeddings", base_url.trim_end_matches('/'));
                let payload = serde_json::json!({
                    "model": model,
                    "input": texts,
                });
                let mut request = ProviderEgressRequest::from_env(
                    "service:embedding",
                    "memory_embedding_batch",
                    "openai_compat",
                    model,
                    "text",
                    &["text", "embedding_input"],
                    payload,
                );
                if let Some(scope) = scope {
                    request = request
                        .with_workspace_scope(&scope.tenant_id, &scope.workspace_id)
                        .with_explicit_user_approval(scope.explicit_provider_approval);
                }
                let resp = self
                    .egress
                    .post_json(&url, Some(api_key), request)
                    .await
                    .map_err(|e| format!("http_error:{e}"))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(format!("provider_error:{status}:{body}"));
                }

                let parsed: EmbeddingResponse =
                    resp.json().await.map_err(|e| format!("parse_error:{e}"))?;
                if parsed.data.is_empty() {
                    return Err("empty_embedding_batch".to_string());
                }
                let mut out = parsed
                    .data
                    .into_iter()
                    .map(|d| normalize_dims(d.embedding, self.dims))
                    .collect::<Vec<_>>();

                if out.len() < texts.len() {
                    out.resize_with(texts.len(), || vec![0.0; self.dims]);
                } else if out.len() > texts.len() {
                    out.truncate(texts.len());
                }
                Ok(out)
            }
        }
    }
}



pub fn vector_literal(vec: &[f32]) -> String {
    let body = vec
        .iter()
        .map(|x| format!("{:.8}", x))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{}]", body)
}

pub fn hash_embed(text: &str, dims: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; dims];
    for tok in text
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
    {
        let mut h: u64 = 1469598103934665603;
        for b in tok.as_bytes() {
            h ^= *b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        let idx = (h % (dims as u64)) as usize;
        v[idx] += 1.0;
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn normalize_dims(mut v: Vec<f32>, dims: usize) -> Vec<f32> {
    if v.len() > dims {
        v.truncate(dims);
    } else if v.len() < dims {
        v.resize(dims, 0.0);
    }

    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}




#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_embed_should_keep_dimension() {
        let v = hash_embed("hello world", 64);
        assert_eq!(v.len(), 64);
    }

    #[test]
    fn vector_literal_should_format_pgvector() {
        let v = vec![0.1_f32, 0.2_f32, 0.3_f32];
        let lit = vector_literal(&v);
        assert!(lit.starts_with('['));
        assert!(lit.ends_with(']'));
        assert!(lit.contains(","));
    }

}
