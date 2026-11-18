//! Shared pieces of the model clients (embeddings, reranking): JSON POST with
//! retries, header and key settings, request templates and reply paths.

use serde_json::Value;
use std::time::Duration;

/// How an API key is sent.
#[derive(Clone, Debug, PartialEq)]
pub enum Auth {
    Bearer,
    None,
    Header(String),
}

impl Auth {
    /// `bearer`, `none` or `header:Name`; `None` when unset.
    pub fn parse(setting: Option<&str>, name: &str) -> Option<Auth> {
        let a = setting?;
        Some(if a.eq_ignore_ascii_case("bearer") {
            Auth::Bearer
        } else if a.eq_ignore_ascii_case("none") {
            Auth::None
        } else {
            match a.split_once(':') {
                Some((kind, header)) if kind.trim().eq_ignore_ascii_case("header") && !header.trim().is_empty() => Auth::Header(header.trim().to_string()),
                _ => panic!("{name} must be bearer, none or header:Name, not {a:?}"),
            }
        })
    }
}

/// Parses "Name: value; Name2: value2" (or one header per line).
pub fn parse_headers(raw: &str) -> Vec<(String, String)> {
    raw.split([';', '\n'])
        .filter_map(|h| h.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .filter(|(k, _)| !k.is_empty())
        .collect()
}

/// The key header (if any) followed by the extra headers.
pub fn headers(key: Option<&str>, auth: &Auth, extra: &[(String, String)]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    if let Some(key) = key {
        match auth {
            Auth::Bearer => out.push(("Authorization".to_string(), format!("Bearer {key}"))),
            Auth::Header(name) => out.push((name.clone(), key.to_string())),
            Auth::None => {}
        }
    }
    out.extend(extra.iter().cloned());
    out
}

pub fn client() -> reqwest::Client {
    reqwest::Client::builder().timeout(Duration::from_secs(120)).build().expect("http client")
}

/// One JSON request, retried on 429, server errors and connection errors.
/// `what` names the call in errors ("embedding", "rerank").
pub async fn post(client: &reqwest::Client, url: &str, headers: &[(String, String)], body: &Value, what: &str) -> Result<Value, String> {
    let mut attempt = 0;
    loop {
        attempt += 1;
        let mut req = client.post(url).json(body);
        for (k, v) in headers {
            req = req.header(k.as_str(), v.as_str());
        }
        match req.send().await {
            Ok(resp) if resp.status().is_success() => return resp.json().await.map_err(|e| format!("{what} reply: {e}")),
            Ok(resp) if attempt < 6 && (resp.status().as_u16() == 429 || resp.status().is_server_error()) => {}
            Ok(resp) => {
                let status = resp.status();
                let text = resp.text().await.unwrap_or_default();
                return Err(format!("{what} HTTP {status}: {}", text.chars().take(300).collect::<String>()));
            }
            Err(e) if attempt < 6 => eprintln!("{what} request error (retrying): {e}"),
            Err(e) => return Err(format!("{what} request: {e}")),
        }
        tokio::time::sleep(Duration::from_millis(500 * (1 << attempt.min(6)))).await;
    }
}

/// The template with placeholders filled: a string that is exactly
/// `"{{name}}"` becomes the value itself (a list, number, …); inside longer
/// strings `{{name}}` is replaced by the value's text (null as empty).
pub fn fill(template: &Value, values: &[(&str, Value)]) -> Value {
    match template {
        Value::String(s) => {
            if let Some((_, v)) = values.iter().find(|(name, _)| s.strip_prefix("{{").and_then(|r| r.strip_suffix("}}")) == Some(*name)) {
                return v.clone();
            }
            let mut out = s.clone();
            for (name, v) in values {
                let text = match v {
                    Value::String(t) => t.clone(),
                    Value::Null => String::new(),
                    other => other.to_string(),
                };
                out = out.replace(&format!("{{{{{name}}}}}"), &text);
            }
            Value::String(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(|v| fill(v, values)).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), fill(v, values))).collect()),
        other => other.clone(),
    }
}

/// Values at a dotted path; `*` takes every element of an array, a number one
/// element (e.g. "data.*.embedding", "results", "output.0.items"). A path that
/// ends on a list of lists or of objects selects its elements.
pub fn select<'a>(value: &'a Value, path: &str) -> Vec<&'a Value> {
    let mut current = vec![value];
    for part in path.split('.').filter(|p| !p.is_empty()) {
        current = current
            .into_iter()
            .flat_map(|v| match (part, v) {
                ("*", Value::Array(a)) => a.iter().collect::<Vec<_>>(),
                (key, Value::Object(o)) => o.get(key).into_iter().collect(),
                (index, Value::Array(a)) => index.parse::<usize>().ok().and_then(|i| a.get(i)).into_iter().collect(),
                _ => Vec::new(),
            })
            .collect();
    }
    if let [Value::Array(a)] = current.as_slice() {
        if a.first().is_some_and(|f| f.is_array() || f.is_object()) {
            return a.iter().collect();
        }
    }
    current
}

/// A setting's value from `get`, trimmed; empty counts as unset.
pub fn setting(get: &impl Fn(&str) -> Option<String>, name: &str) -> Option<String> {
    get(name).map(|v| v.trim().to_string()).filter(|v| !v.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn select_follows_paths() {
        let v = json!({"data": [{"embedding": [1]}, {"embedding": [2]}], "nested": {"list": [[3], [4]]}, "results": [{"index": 0}]});
        assert_eq!(select(&v, "data.*.embedding"), vec![&json!([1]), &json!([2])]);
        assert_eq!(select(&v, "nested.list"), vec![&json!([3]), &json!([4])]);
        assert_eq!(select(&v, "data.1.embedding"), vec![&json!([2])]);
        assert_eq!(select(&v, "results"), vec![&json!({"index": 0})]);
        assert!(select(&v, "missing.path").is_empty());
    }

    #[test]
    fn fill_replaces_whole_values_and_text() {
        let t = json!({"items": "{{inputs}}", "label": "model {{model}} at {{dims}}", "size": "{{dims}}", "none": "{{missing}}x{{nothing}}"});
        let out = fill(&t, &[("inputs", json!(["a", "b"])), ("model", json!("m")), ("dims", json!(64)), ("nothing", Value::Null)]);
        assert_eq!(out, json!({"items": ["a", "b"], "label": "model m at 64", "size": 64, "none": "{{missing}}x"}));
    }
}
