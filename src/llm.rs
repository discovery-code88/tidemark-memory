//! The optional chat model (`TIDEMARK_LLM_*`), for features that need one;
//! Add and Search never call it. Any API: OpenAI-compatible, Anthropic, Gemini,
//! a custom JSON template or a local command-line tool (see
//! `tidemark_engine::chat` for every setting).

use std::sync::Arc;
use tidemark_engine::ChatModel;

pub const PREFIX: &str = "TIDEMARK_LLM";

/// The configured model, checked with one short call at start (a wrong key or
/// URL refuses the start; `TIDEMARK_SKIP_LLM_CHECK=1` skips the call).
pub async fn from_env() -> Result<Option<Arc<ChatModel>>, String> {
    let Some(model) = ChatModel::from_env(PREFIX)? else { return Ok(None) };
    if std::env::var_os("TIDEMARK_SKIP_LLM_CHECK").is_none() {
        model
            .complete("Reply with the single word OK.", "Are you there?")
            .await
            .map_err(|e| format!("language model check failed ({}): {e}", model.label()))?;
    }
    Ok(Some(Arc::new(model)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};
    use std::sync::Mutex;

    type Seen = Arc<Mutex<Vec<(String, std::collections::HashMap<String, String>, Value)>>>;

    /// Answers like OpenAI, Anthropic or Gemini depending on the path; the
    /// first call fails with 503 when `flaky`.
    async fn mock(flaky: bool) -> (String, Seen) {
        let seen: Seen = Arc::default();
        let log = seen.clone();
        let app = axum::Router::new().fallback(move |uri: axum::http::Uri, headers: axum::http::HeaderMap, axum::Json(body): axum::Json<Value>| {
            let log = log.clone();
            async move {
                let path = uri.path().to_string();
                let h = headers.iter().map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string())).collect();
                let first = { let mut l = log.lock().unwrap(); l.push((path.clone(), h, body)); l.len() == 1 };
                if flaky && first {
                    return (axum::http::StatusCode::SERVICE_UNAVAILABLE, axum::Json(json!({"error": "busy"})));
                }
                let reply = if path.ends_with("/chat/completions") {
                    json!({"choices": [{"message": {"content": "{\"answer\": \"openai\"}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
                } else if path.ends_with("/messages") {
                    json!({"content": [{"type": "text", "text": "{\"answer\": \"anthropic\"}"}], "stop_reason": "end_turn", "usage": {"input_tokens": 10, "output_tokens": 3}})
                } else {
                    json!({"candidates": [{"content": {"parts": [{"text": "{\"answer\": \"gemini\"}"}]}, "finishReason": "STOP"}], "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3}})
                };
                (axum::http::StatusCode::OK, axum::Json(reply))
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (url, seen)
    }

    fn model(prefix: &str, vars: &[(&str, &str)]) -> ChatModel {
        for (k, v) in vars {
            std::env::set_var(format!("{prefix}_{k}"), v);
        }
        ChatModel::from_env(prefix).unwrap().expect("configured")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn every_api_answers_in_json_with_usage_counted() {
        for (protocol, path, auth) in [("openai", "/v1/chat/completions", "authorization"), ("anthropic", "/v1/messages", "x-api-key"), ("gemini", "/v1beta/models/m1:generateContent", "x-goog-api-key")] {
            let (url, seen) = mock(false).await;
            let base = if protocol == "gemini" { format!("{url}/v1beta") } else { format!("{url}/v1") };
            let prefix = format!("LLM_TEST_{}", protocol.to_uppercase());
            let m = model(&prefix, &[("PROTOCOL", protocol), ("URL", &base), ("MODEL", "m1"), ("API_KEY", "secret")]);
            assert_eq!(m.complete_json("sys", "question", None).await.unwrap(), json!({"answer": protocol}), "{protocol}");
            let usage = m.usage();
            assert_eq!((usage.calls, usage.input_tokens, usage.output_tokens), (1, 10, 3), "{protocol}");
            let log = seen.lock().unwrap().clone();
            assert_eq!(log[0].0, path, "{protocol}");
            assert!(log[0].1.get(auth).is_some_and(|v| v.contains("secret")), "{protocol}: key in {auth}");
            assert!(log[0].2.to_string().contains("question"), "{protocol}");
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_busy_server_is_retried() {
        let (url, seen) = mock(true).await;
        let m = model("LLM_TEST_RETRY", &[("PROTOCOL", "openai"), ("URL", &format!("{url}/v1")), ("MODEL", "m1")]);
        assert_eq!(m.complete("sys", "q").await.unwrap(), "{\"answer\": \"openai\"}");
        assert_eq!(seen.lock().unwrap().len(), 2);
        let (url, _) = mock(true).await;
        let m = model("LLM_TEST_NORETRY", &[("PROTOCOL", "openai"), ("URL", &format!("{url}/v1")), ("MODEL", "m1"), ("RETRIES", "0")]);
        assert_eq!(m.complete("sys", "q").await.unwrap_err(), "llm_http_status:503");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn a_custom_template_reaches_any_api() {
        let (url, seen) = mock(false).await;
        let m = model("LLM_TEST_CUSTOM", &[
            ("PROTOCOL", "custom"),
            ("URL", &format!("{url}/v1/chat/completions")),
            ("REQUEST_TEMPLATE", r#"{"prompt": "{{system}} | {{user}}", "limit": "{{max_tokens}}"}"#),
            ("RESPONSE_PATH", "/choices/0/message/content"),
            ("USAGE_INPUT_PATH", "/usage/prompt_tokens"),
        ]);
        assert_eq!(m.complete("sys", "q").await.unwrap(), "{\"answer\": \"openai\"}");
        assert_eq!(seen.lock().unwrap()[0].2, json!({"prompt": "sys | q", "limit": 4096}));
        assert_eq!(m.usage().input_tokens, 10);
    }
}
