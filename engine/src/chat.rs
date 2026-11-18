//! A chat model for hosts of the engine, in whatever API the user has:
//! OpenAI-compatible, Anthropic, Gemini, a custom JSON template, or a local
//! command-line tool (see [`crate::llm_protocol`]). Configured from
//! environment variables under one prefix, e.g. `TIDEMARK_LLM`:
//!
//! - `{P}_URL` (not needed for a command), `{P}_MODEL`, `{P}_API_KEY`;
//! - `{P}_PROTOCOL`: openai | anthropic | gemini | custom | command (default:
//!   detected from the URL); `{P}_HEADERS` (`Name: value; …`, `{{api_key}}`
//!   replaces the default auth header);
//! - custom: `{P}_REQUEST_TEMPLATE`, `{P}_RESPONSE_PATH`; command:
//!   `{P}_COMMAND`, `{P}_COMMAND_CWD`;
//! - `{P}_MAX_TOKENS` (default 4096), `{P}_TEMPERATURE` (0), `{P}_RETRIES`
//!   (transient failures retried, default 3), `{P}_NATIVE_JSON` (default on:
//!   use the API's JSON mode where it has one, falling back automatically);
//! - `{P}_MAX_ATTEMPTS`: operator cap on attempts (every try, retries included) since start; beyond it
//!   calls fail with `llm_attempt_cap_reached` without being sent (unset = no cap);
//! - `{P}_USAGE_INPUT_PATH`, `_USAGE_OUTPUT_PATH`, `_USAGE_CACHED_PATH`,
//!   `_USAGE_REGEX`: where token counts are for custom APIs and commands.
//!
//! The request timeout is `PROVIDER_EGRESS_REQUEST_TIMEOUT_MS` (default 30 s).

use crate::llm_protocol::{extract_json, LlmCall, LlmEndpoint, LlmRequest, TokenUsage, UsagePaths};
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct ChatModel {
    endpoint: LlmEndpoint,
    model: String,
    key: Option<String>,
    max_tokens: u32,
    temperature: f32,
    retries: u32,
    native_json: bool,
    usage_paths: UsagePaths,
    max_attempts: Option<u64>,
    attempts: AtomicU64,
    calls: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
}

/// Calls and tokens since start.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ChatUsage {
    /// Requests tried, retries included.
    pub attempts: u64,
    pub calls: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
}

impl ChatModel {
    /// `None` when neither `{prefix}_URL` nor a command protocol is set.
    pub fn from_env(prefix: &str) -> Result<Option<Self>, String> {
        let var = |name: &str| std::env::var(format!("{prefix}_{name}")).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let command = var("PROTOCOL").is_some_and(|p| p.eq_ignore_ascii_case("command"));
        let Some(url) = var("URL").or_else(|| command.then(String::new)) else {
            return Ok(None);
        };
        let endpoint = LlmEndpoint::from_env(prefix, url.trim_end_matches('/'))?;
        let model = match var("MODEL") {
            Some(m) => m,
            None if command || endpoint.protocol_name() == "custom" => String::new(),
            None => return Err(format!("{prefix}_MODEL is required")),
        };
        let number = |name: &str, default: f64| -> Result<f64, String> {
            var(name).map_or(Ok(default), |v| v.parse::<f64>().map_err(|_| format!("{prefix}_{name} must be a number, not {v:?}")))
        };
        Ok(Some(Self {
            endpoint,
            model,
            key: var("API_KEY"),
            max_tokens: number("MAX_TOKENS", 4096.0)?.max(1.0) as u32,
            temperature: number("TEMPERATURE", 0.0)? as f32,
            retries: number("RETRIES", 3.0)?.max(0.0) as u32,
            native_json: !var("NATIVE_JSON").is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "0" | "off" | "false" | "no")),
            usage_paths: UsagePaths::from_env(prefix),
            max_attempts: match var("MAX_ATTEMPTS") {
                Some(v) => Some(v.parse::<u64>().map_err(|_| format!("{prefix}_MAX_ATTEMPTS must be a whole number, not {v:?}"))?),
                None => None,
            },
            attempts: AtomicU64::new(0),
            calls: AtomicU64::new(0),
            input_tokens: AtomicU64::new(0),
            output_tokens: AtomicU64::new(0),
        }))
    }

    /// "protocol:model", e.g. "anthropic:claude-haiku-4-5" (just the protocol without a model).
    pub fn label(&self) -> String {
        if self.model.is_empty() { self.endpoint.protocol_name().to_string() } else { format!("{}:{}", self.endpoint.protocol_name(), self.model) }
    }

    pub fn usage(&self) -> ChatUsage {
        ChatUsage {
            attempts: self.attempts.load(Ordering::Relaxed),
            calls: self.calls.load(Ordering::Relaxed),
            input_tokens: self.input_tokens.load(Ordering::Relaxed),
            output_tokens: self.output_tokens.load(Ordering::Relaxed),
        }
    }

    /// The reply text to `system` + `user`.
    pub async fn complete(&self, system: &str, user: &str) -> Result<String, String> {
        self.call(system, user, false, None).await
    }

    /// A reply that must be one JSON object (with `schema`, matching it where
    /// the API enforces schemas; the caller still validates it).
    pub async fn complete_json(&self, system: &str, user: &str, schema: Option<(&str, Value)>) -> Result<Value, String> {
        let text = self.call(system, user, true, schema).await?;
        extract_json(&text).map_err(|e| format!("llm_reply_not_json:{e}"))
    }

    async fn call(&self, system: &str, user: &str, json: bool, schema: Option<(&str, Value)>) -> Result<String, String> {
        let call = LlmCall {
            endpoint: &self.endpoint,
            api_key: self.key.as_deref(),
            service: "service:host",
            purpose: "host_model_call",
            data_classes: &["text", "memory_evidence"],
            tenant_id: "host",
            workspace_id: "default",
            explicit_approval: true,
        };
        let mut attempt = 0;
        loop {
            // The cap counts every try; a reserved try that would exceed it is not sent.
            if let Some(max) = self.max_attempts {
                if self.attempts.fetch_add(1, Ordering::SeqCst) >= max {
                    self.attempts.fetch_sub(1, Ordering::SeqCst);
                    return Err("llm_attempt_cap_reached".into());
                }
            } else {
                self.attempts.fetch_add(1, Ordering::Relaxed);
            }
            let request = LlmRequest {
                model: &self.model,
                system,
                user: user.to_string(),
                temperature: self.temperature,
                json,
                schema: schema.clone(),
                strict: schema.is_some(),
                native_json: self.native_json,
                max_tokens: self.max_tokens,
            };
            match call.complete_with_usage(&request, &self.usage_paths).await {
                Ok((text, usage)) => {
                    self.calls.fetch_add(1, Ordering::Relaxed);
                    let usage = usage.unwrap_or(TokenUsage::default());
                    self.input_tokens.fetch_add(usage.input, Ordering::Relaxed);
                    self.output_tokens.fetch_add(usage.output, Ordering::Relaxed);
                    return Ok(text);
                }
                Err(e) if attempt < self.retries && transient(&e) => {
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_millis(500 * (1 << attempt.min(6)))).await;
                }
                Err(e) => return Err(e),
            }
        }
    }
}

/// Rate limits, server errors and connection failures are worth retrying;
/// a rejected request or a bad key is not.
fn transient(error: &str) -> bool {
    match error.strip_prefix("llm_http_status:").and_then(|s| s.parse::<u16>().ok()) {
        Some(status) => status == 408 || status == 429 || status >= 500,
        None => error.starts_with("provider_egress_http_error"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn only_transient_failures_are_retried() {
        assert!(transient("llm_http_status:429") && transient("llm_http_status:503") && transient("provider_egress_http_error:timeout"));
        assert!(!transient("llm_http_status:401") && !transient("llm_http_status:400") && !transient("llm_content_missing"));
    }

    #[tokio::test]
    async fn a_command_model_answers_in_json_and_counts_usage() {
        let prefix = "CHAT_TEST_CMD";
        std::env::set_var(format!("{prefix}_PROTOCOL"), "command");
        std::env::set_var(format!("{prefix}_COMMAND"), "sh -c 'cat >/dev/null; echo \"tokens used: 12\" >&2; printf \"%s\" \"{\\\"ok\\\": true}\"'");
        std::env::set_var(format!("{prefix}_USAGE_REGEX"), r"tokens used: (?P<total>\d+)");
        let model = ChatModel::from_env(prefix).unwrap().expect("configured");
        assert_eq!(model.label(), "command");
        assert_eq!(model.complete_json("sys", "user", None).await.unwrap(), json!({"ok": true}));
        assert_eq!(model.usage(), ChatUsage { attempts: 1, calls: 1, input_tokens: 12, output_tokens: 0 });
    }

    #[tokio::test]
    async fn the_attempt_cap_counts_every_try_and_stops_sending() {
        let prefix = "CHAT_TEST_CAP";
        std::env::set_var(format!("{prefix}_PROTOCOL"), "command");
        std::env::set_var(format!("{prefix}_COMMAND"), "sh -c 'cat >/dev/null; echo ok'");
        std::env::set_var(format!("{prefix}_MAX_ATTEMPTS"), "2");
        let model = ChatModel::from_env(prefix).unwrap().unwrap();
        assert!(model.complete("s", "u").await.is_ok() && model.complete("s", "u").await.is_ok());
        assert_eq!(model.complete("s", "u").await.unwrap_err(), "llm_attempt_cap_reached");
        assert_eq!((model.usage().attempts, model.usage().calls), (2, 2));
    }

    #[test]
    fn unset_means_no_model_and_a_url_needs_a_model() {
        assert!(ChatModel::from_env("CHAT_TEST_UNSET").unwrap().is_none());
        std::env::set_var("CHAT_TEST_NOMODEL_URL", "http://127.0.0.1:1/v1");
        assert!(ChatModel::from_env("CHAT_TEST_NOMODEL").is_err());
    }
}
