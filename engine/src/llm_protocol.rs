//! Chat-model wire protocols. Users bring whichever model API they have, so
//! every LLM caller builds one `LlmRequest` and the protocol adapter turns it
//! into that API's request and reads the reply text back:
//!
//! - `openai`: OpenAI Chat Completions and everything compatible with it
//!   (Ollama, llama.cpp, vLLM, LM Studio, OpenRouter, Gemini's compatibility
//!   endpoint, ...);
//! - `anthropic`: Anthropic Messages API;
//! - `gemini`: Google Gemini `generateContent`;
//! - `command`: a local command-line model tool (Codex CLI, `llama-cli`,
//!   `ollama run`, ...): the prompt goes to its stdin, the reply comes from
//!   its stdout or from a file it writes. No shell is involved.
//!
//! Structured output degrades by capability: a strict JSON schema where the
//! API enforces one, otherwise JSON mode with the schema stated in the system
//! prompt. Callers validate every reply strictly either way.

use serde_json::{json, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LlmProtocol {
    OpenAi,
    Anthropic,
    Gemini,
}

/// Anthropic Messages API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub(crate) struct LlmRequest<'a> {
    pub model: &'a str,
    pub system: &'a str,
    pub user: String,
    pub temperature: f32,
    /// Reply must be one JSON object; with a schema, matching it.
    pub json: bool,
    /// (name, JSON schema) for APIs that enforce schemas.
    pub schema: Option<(&'a str, Value)>,
    /// Use the API's strict schema enforcement when it has one.
    pub strict: bool,
    /// Send the API's JSON parameter (`response_format`, `responseMimeType`,
    /// `output_config`). Off for servers that reject it: the expected JSON is
    /// then requested in the instructions only.
    pub native_json: bool,
    pub max_tokens: u32,
}

impl LlmProtocol {
    /// Explicit `openai` | `anthropic` | `gemini`, else detected from the URL.
    pub(crate) fn resolve(explicit: Option<&str>, url: &str) -> Result<Self, String> {
        match explicit.map(|value| value.trim().to_ascii_lowercase()).as_deref() {
            Some("openai") | Some("openai_compat") => Ok(Self::OpenAi),
            Some("anthropic") => Ok(Self::Anthropic),
            Some("gemini") | Some("google") => Ok(Self::Gemini),
            Some("") | None => Ok(Self::detect(url)),
            Some(other) => Err(format!("llm_protocol_unknown:{other}")),
        }
    }

    fn detect(url: &str) -> Self {
        let lower = url.to_ascii_lowercase();
        if lower.contains("api.anthropic.com") || lower.ends_with("/v1/messages") {
            Self::Anthropic
        } else if lower.contains("generativelanguage.googleapis.com") && !lower.contains("/openai") {
            Self::Gemini
        } else {
            Self::OpenAi
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai_compat",
            Self::Anthropic => "anthropic",
            Self::Gemini => "gemini",
        }
    }

    /// The request URL. A full endpoint URL is used as given; a base URL gets
    /// the protocol's path (`/v1/messages`, `/models/{model}:generateContent`).
    pub(crate) fn endpoint(self, url: &str, model: &str) -> String {
        let base = url.trim_end_matches('/');
        match self {
            // A base ending in a version segment (`/v1`, `/v1beta`, `/openai`)
            // gets the Chat Completions path, before any query string; any
            // other URL is a full endpoint and used as given.
            Self::OpenAi => {
                let (path, query) = base.split_once('?').map_or((base, None), |(path, query)| (path, Some(query)));
                let path = path.trim_end_matches('/');
                let last = path.rsplit('/').next().unwrap_or("");
                let is_base = last == "openai" || (last.starts_with('v') && last[1..].starts_with(|c: char| c.is_ascii_digit()));
                match (is_base, query) {
                    (true, Some(query)) => format!("{path}/chat/completions?{query}"),
                    (true, None) => format!("{path}/chat/completions"),
                    (false, _) => base.to_string(),
                }
            }
            Self::Anthropic if base.ends_with("/messages") => base.to_string(),
            Self::Anthropic if base.ends_with("/v1") => format!("{base}/messages"),
            Self::Anthropic => format!("{base}/v1/messages"),
            Self::Gemini if base.contains(":generateContent") => base.to_string(),
            Self::Gemini => format!("{base}/models/{model}:generateContent"),
        }
    }

    /// Protocol auth headers for an API key.
    pub(crate) fn auth_headers(self, key: Option<&str>) -> Vec<(String, String)> {
        let key = key.filter(|value| !value.is_empty());
        match self {
            Self::OpenAi => key.map(|key| vec![("authorization".into(), format!("Bearer {key}"))]).unwrap_or_default(),
            Self::Anthropic => {
                let mut headers = vec![("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string())];
                if let Some(key) = key {
                    headers.push(("x-api-key".into(), key.into()));
                }
                headers
            }
            Self::Gemini => key.map(|key| vec![("x-goog-api-key".into(), key.into())]).unwrap_or_default(),
        }
    }

    /// The system instructions as sent: without an enforced schema the schema
    /// is stated in them; OpenAI JSON mode keeps the caller's own instructions.
    pub(crate) fn instructions(self, request: &LlmRequest) -> String {
        let native = request.json && request.native_json;
        let enforced = native && request.strict && matches!(self, Self::OpenAi | Self::Anthropic);
        match (&request.schema, enforced) {
            (Some((_, schema)), false) if request.json && (self != Self::OpenAi || !native) => format!(
                "{}\nReturn only one JSON object matching this JSON Schema exactly, with no other text:\n{schema}",
                request.system
            ),
            (None, _) if request.json && !native => format!(
                "{}\nReturn only one JSON object, with no other text.", request.system
            ),
            _ => request.system.to_string(),
        }
    }

    pub(crate) fn body(self, request: &LlmRequest) -> Value {
        let native = request.json && request.native_json;
        let enforced = native && request.strict && matches!(self, Self::OpenAi | Self::Anthropic);
        let system = self.instructions(request);
        match self {
            Self::OpenAi => {
                let mut body = json!({"model":request.model,"temperature":request.temperature,"messages":[
                    {"role":"system","content":system},{"role":"user","content":request.user}]});
                if native {
                    body["response_format"] = match (&request.schema, enforced) {
                        (Some((name, schema)), true) => json!({"type":"json_schema",
                            "json_schema":{"name":name,"strict":true,"schema":schema}}),
                        _ => json!({"type":"json_object"}),
                    };
                }
                body
            }
            Self::Anthropic => {
                let mut body = json!({"model":request.model,"max_tokens":request.max_tokens,
                    "temperature":request.temperature,"system":system,
                    "messages":[{"role":"user","content":request.user}]});
                if let (true, Some((_, schema))) = (enforced, &request.schema) {
                    body["output_config"] = json!({"format":{"type":"json_schema","schema":schema}});
                }
                body
            }
            Self::Gemini => {
                let mut config = json!({"temperature":request.temperature,"maxOutputTokens":request.max_tokens});
                if native {
                    config["responseMimeType"] = json!("application/json");
                }
                json!({"systemInstruction":{"parts":[{"text":system}]},
                    "contents":[{"role":"user","parts":[{"text":request.user}]}],
                    "generationConfig":config})
            }
        }
    }

    /// The reply text and whether it was cut off at the reply budget (so it
    /// can be continued); refusals and blocks stay errors.
    pub(crate) fn extract_text_status(self, payload: &Value) -> Result<(String, bool), String> {
        let truncated = match self {
            Self::OpenAi => payload.pointer("/choices/0/finish_reason").and_then(Value::as_str) == Some("length"),
            Self::Anthropic => payload.get("stop_reason").and_then(Value::as_str) == Some("max_tokens"),
            Self::Gemini => payload.pointer("/candidates/0/finishReason").and_then(Value::as_str) == Some("MAX_TOKENS"),
        };
        if !truncated {
            return self.extract_text(payload).map(|text| (text, false));
        }
        let text = match self {
            Self::OpenAi => payload.pointer("/choices/0/message/content").and_then(Value::as_str).unwrap_or("").to_string(),
            Self::Anthropic => payload.get("content").and_then(Value::as_array).into_iter().flatten()
                .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|block| block.get("text").and_then(Value::as_str)).collect(),
            Self::Gemini => payload.pointer("/candidates/0/content/parts").and_then(Value::as_array).into_iter().flatten()
                .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
                .filter_map(|part| part.get("text").and_then(Value::as_str)).collect(),
        };
        Ok((text, true))
    }

    /// The reply text, or why there is none (truncated, refused, blocked).
    pub(crate) fn extract_text(self, payload: &Value) -> Result<String, String> {
        match self {
            Self::OpenAi => {
                if let Some(reason) = payload.pointer("/choices/0/finish_reason").and_then(Value::as_str) {
                    if reason != "stop" {
                        return Err(format!("llm_generation_incomplete:{reason}"));
                    }
                }
                payload.pointer("/choices/0/message/content").and_then(Value::as_str)
                    .map(str::to_string).ok_or_else(|| "llm_content_missing".into())
            }
            Self::Anthropic => {
                match payload.get("stop_reason").and_then(Value::as_str) {
                    None | Some("end_turn") | Some("stop_sequence") => {}
                    Some(reason) => return Err(format!("llm_generation_incomplete:{reason}")),
                }
                let text = payload.get("content").and_then(Value::as_array).into_iter().flatten()
                    .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                    .filter_map(|block| block.get("text").and_then(Value::as_str))
                    .collect::<String>();
                if text.is_empty() { Err("llm_content_missing".into()) } else { Ok(text) }
            }
            Self::Gemini => {
                let candidate = payload.pointer("/candidates/0").ok_or("llm_content_missing")?;
                match candidate.get("finishReason").and_then(Value::as_str) {
                    None | Some("STOP") => {}
                    Some(reason) => return Err(format!("llm_generation_incomplete:{reason}")),
                }
                // Thinking models may return thought parts; only answer parts count.
                let text = candidate.pointer("/content/parts").and_then(Value::as_array).into_iter().flatten()
                    .filter(|part| part.get("thought").and_then(Value::as_bool) != Some(true))
                    .filter_map(|part| part.get("text").and_then(Value::as_str))
                    .collect::<String>();
                if text.is_empty() { Err("llm_content_missing".into()) } else { Ok(text) }
            }
        }
    }
}

/// The JSON object in a model reply: the whole reply, a fenced block, or the
/// first balanced `{...}` amid prose. Validation of its content stays with
/// the caller.
pub(crate) fn extract_json(text: &str) -> Result<Value, String> {
    let trimmed = text.trim();
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Ok(value);
    }
    let unfenced = trimmed.trim_start_matches("```json").trim_start_matches("```JSON")
        .trim_start_matches("```").trim_end_matches("```").trim();
    if let Ok(value) = serde_json::from_str::<Value>(unfenced) {
        return Ok(value);
    }
    let bytes = trimmed.as_bytes();
    for (start, _) in trimmed.match_indices('{') {
        let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
        for (offset, &byte) in bytes[start..].iter().enumerate() {
            match byte {
                _ if escaped => escaped = false,
                b'\\' if in_string => escaped = true,
                b'"' => in_string = !in_string,
                b'{' if !in_string => depth += 1,
                b'}' if !in_string => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(value) = serde_json::from_str::<Value>(&trimmed[start..=start + offset]) {
                            return Ok(value);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    Err("llm_reply_has_no_json_object".into())
}

/// A 400/422 reply that names the JSON-output parameter: the server does not
/// support native JSON output, so the call is retried without it.
pub(crate) fn rejects_native_json(status: u16, body: &Value) -> bool {
    let text = body.to_string().to_ascii_lowercase();
    matches!(status, 400 | 422)
        && ["response_format", "json_schema", "json_object", "responsemimetype", "output_config", "json mode"]
            .iter()
            .any(|marker| text.contains(marker))
}

static NO_NATIVE_JSON: std::sync::OnceLock<std::sync::Mutex<std::collections::HashSet<String>>> =
    std::sync::OnceLock::new();

/// Whether native JSON output may be requested from this endpoint.
pub(crate) fn native_json_allowed(endpoint: &str) -> bool {
    NO_NATIVE_JSON.get_or_init(Default::default).lock()
        .map(|set| !set.contains(endpoint)).unwrap_or(true)
}

/// Remembers (for this process) that an endpoint rejected native JSON output.
pub(crate) fn remember_no_native_json(endpoint: &str) {
    if let Ok(mut set) = NO_NATIVE_JSON.get_or_init(Default::default).lock() {
        set.insert(endpoint.to_string());
    }
}

/// A configured model endpoint: protocol, URL and any header or request
/// overrides, read from `{PREFIX}_PROTOCOL`, `{PREFIX}_HEADERS`,
/// `{PREFIX}_REQUEST_TEMPLATE` and `{PREFIX}_RESPONSE_PATH`. This covers APIs
/// without a built-in adapter:
///
/// - `{PREFIX}_HEADERS`: `Name: value` pairs separated by `;`, applied to any
///   protocol; `{{api_key}}` is replaced by the key, and a header using it
///   replaces the protocol's default auth header (e.g. Azure `api-key`).
/// - `{PREFIX}_PROTOCOL=custom` with `{PREFIX}_REQUEST_TEMPLATE`: a JSON body
///   whose strings may contain `{{model}}`, `{{system}}`, `{{user}}`; a string
///   that is exactly `{{max_tokens}}` or `{{temperature}}` becomes a number.
///   `{PREFIX}_RESPONSE_PATH` is a JSON pointer to the reply text.
/// - `{{model}}` in the URL is replaced by the model (deployment paths).
#[derive(Debug, Clone)]
pub(crate) struct LlmEndpoint {
    pub protocol: LlmProtocol,
    pub url: String,
    headers: Vec<(String, String)>,
    custom: Option<(Value, String)>,
    /// `{PREFIX}_PROTOCOL=command`: the argv of a local model tool.
    command: Option<Vec<String>>,
    /// Custom templates: JSON pointer to the reply's finish reason and the
    /// values that mean "cut off at the budget" (`{PREFIX}_FINISH_PATH`,
    /// `{PREFIX}_TRUNCATED_VALUES`, default "length,max_tokens,MAX_TOKENS").
    finish: Option<(String, Vec<String>)>,
    /// `{PREFIX}_COMMAND_CWD`: the working directory of a command tool. Unset,
    /// each call runs in a fresh empty directory, so an agent-style tool
    /// (Codex) cannot see the project or stored results.
    command_cwd: Option<String>,
}

impl LlmEndpoint {
    pub(crate) fn from_env(prefix: &str, url: &str) -> Result<Self, String> {
        let var = |name: &str| std::env::var(format!("{prefix}_{name}")).ok().filter(|value| !value.trim().is_empty());
        let protocol_name = var("PROTOCOL");
        let headers = var("HEADERS").map(|spec| parse_headers(&spec)).transpose()?.unwrap_or_default();
        if protocol_name.as_deref().map(str::trim).is_some_and(|name| name.eq_ignore_ascii_case("custom")) {
            let template = var("REQUEST_TEMPLATE").ok_or(format!("{prefix}_REQUEST_TEMPLATE is required for the custom protocol"))?;
            let template: Value = serde_json::from_str(&template)
                .map_err(|error| format!("{prefix}_REQUEST_TEMPLATE is not JSON: {error}"))?;
            let path = var("RESPONSE_PATH").ok_or(format!("{prefix}_RESPONSE_PATH is required for the custom protocol"))?;
            if !path.starts_with('/') {
                return Err(format!("{prefix}_RESPONSE_PATH must be a JSON pointer such as /output/text"));
            }
            let finish = var("FINISH_PATH").filter(|p| p.starts_with('/')).map(|pointer| (pointer, var("TRUNCATED_VALUES")
                .unwrap_or_else(|| "length,max_tokens,MAX_TOKENS".into()).split(',').map(|v| v.trim().to_string()).filter(|v| !v.is_empty()).collect()));
            return Ok(Self { protocol: LlmProtocol::OpenAi, url: url.to_string(), headers, custom: Some((template, path)), command: None, finish, command_cwd: None });
        }
        if protocol_name.as_deref().map(str::trim).is_some_and(|name| name.eq_ignore_ascii_case("command")) {
            let spec = var("COMMAND").ok_or(format!("{prefix}_COMMAND is required for the command protocol"))?;
            let argv = split_command(&spec)?;
            let command_cwd = std::env::var(format!("{prefix}_COMMAND_CWD")).ok().filter(|dir| !dir.trim().is_empty());
            return Ok(Self { protocol: LlmProtocol::OpenAi, url: url.to_string(), headers, custom: None, command: Some(argv), finish: None, command_cwd });
        }
        Ok(Self { protocol: LlmProtocol::resolve(protocol_name.as_deref(), url)?, url: url.to_string(), headers, custom: None, command: None, finish: None, command_cwd: None })
    }

    pub(crate) fn is_command(&self) -> bool {
        self.command.is_some()
    }

    /// Runs the local model command: `{{model}}` in an argument becomes the
    /// model, `{{output_file}}` a fresh file the tool writes its reply to
    /// (otherwise the reply is its stdout). The prompt is written to stdin.
    /// Bounded by `LLM_COMMAND_CONCURRENCY` (default 2) parallel processes;
    /// a process still running when the caller gives up is killed.
    pub(crate) async fn run_command(&self, model: &str, prompt: &str) -> Result<String, String> {
        self.run_command_inner(model, prompt).await.map(|(reply, _)| reply)
    }

    async fn run_command_inner(&self, model: &str, prompt: &str) -> Result<(String, String), String> {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let argv = self.command.as_ref().ok_or("llm_command_not_configured")?;
        static SLOTS: std::sync::OnceLock<tokio::sync::Semaphore> = std::sync::OnceLock::new();
        let slots = SLOTS.get_or_init(|| tokio::sync::Semaphore::new(
            std::env::var("LLM_COMMAND_CONCURRENCY").ok().and_then(|v| v.parse().ok()).filter(|n: &usize| *n > 0).unwrap_or(2)));
        let _slot = slots.acquire().await.map_err(|_| "llm_command_unavailable")?;
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let output_file = argv.iter().any(|arg| arg.contains("{{output_file}}")).then(|| std::env::temp_dir().join(format!(
            "umem-llm-reply-{}-{}.txt", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))));
        let fill = |arg: &String| {
            let arg = arg.replace("{{model}}", model);
            match &output_file { Some(path) => arg.replace("{{output_file}}", &path.to_string_lossy()), None => arg }
        };
        // An isolated working directory: the tool sees only the prompt.
        let fresh = self.command_cwd.is_none().then(|| std::env::temp_dir().join(format!(
            "umem-llm-cwd-{}-{}", std::process::id(), NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))));
        if let Some(dir) = &fresh {
            std::fs::create_dir_all(dir).map_err(|error| format!("llm_command_cwd_failed:{error}"))?;
        }
        struct RemoveDir(Option<std::path::PathBuf>);
        impl Drop for RemoveDir { fn drop(&mut self) { if let Some(dir) = &self.0 { let _ = std::fs::remove_dir_all(dir); } } }
        let _cleanup = RemoveDir(fresh.clone());
        let cwd = fresh.clone().unwrap_or_else(|| std::path::PathBuf::from(self.command_cwd.clone().unwrap_or_default()));
        let mut child = tokio::process::Command::new(fill(&argv[0]))
            .args(argv[1..].iter().map(fill))
            .current_dir(&cwd)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|error| format!("llm_command_spawn_failed:{error}"))?;
        let mut stdin = child.stdin.take().ok_or("llm_command_no_stdin")?;
        let prompt = prompt.to_string();
        let writer = tokio::spawn(async move { let _ = stdin.write_all(prompt.as_bytes()).await; });
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let mut out = child.stdout.take().ok_or("llm_command_no_stdout")?.take(4 * 1024 * 1024);
        let mut err = child.stderr.take().ok_or("llm_command_no_stderr")?.take(256 * 1024);
        let (read_out, read_err) = tokio::join!(out.read_to_end(&mut stdout), err.read_to_end(&mut stderr));
        read_out.map_err(|error| format!("llm_command_read_failed:{error}"))?;
        let _ = read_err;
        let _ = writer.await;
        let status = child.wait().await.map_err(|error| format!("llm_command_wait_failed:{error}"))?;
        let reply = match &output_file {
            Some(path) => {
                let text = std::fs::read_to_string(path).unwrap_or_default();
                let _ = std::fs::remove_file(path);
                text
            }
            None => String::from_utf8_lossy(&stdout).into_owned(),
        };
        if !status.success() {
            let tail = String::from_utf8_lossy(&stderr);
            let tail = tail.chars().rev().take(300).collect::<String>().chars().rev().collect::<String>();
            return Err(format!("llm_command_failed:{status}:{}", tail.trim()));
        }
        let reply = reply.trim().to_string();
        if reply.is_empty() {
            return Err("llm_command_empty_reply".into());
        }
        let log = format!("{}\n{}", String::from_utf8_lossy(&stderr), String::from_utf8_lossy(&stdout));
        Ok((reply, log))
    }

    /// `run_command` that also returns what the command wrote to stderr and
    /// stdout (bounded), for reading a usage report from it.
    pub(crate) async fn run_command_logged(&self, model: &str, prompt: &str) -> Result<(String, String), String> {
        self.run_command_inner(model, prompt).await
    }

    /// The whole prompt for a command: instructions (with the JSON schema
    /// stated, as for any server without native JSON), then the input.
    pub(crate) fn command_prompt(&self, request: &LlmRequest) -> String {
        let request = LlmRequest { native_json: false, user: request.user.clone(), schema: request.schema.clone(), ..*request };
        format!("Answer using only the instructions and input in this message. Do not run commands, read or search files, \
            or browse the web.\n\n{}\n\n{}", self.protocol.instructions(&request), request.user)
    }

    /// A built-in protocol with no overrides.
    

    pub(crate) fn protocol_name(&self) -> &'static str {
        if self.command.is_some() { "command" } else if self.custom.is_some() { "custom" } else { self.protocol.as_str() }
    }

    pub(crate) fn endpoint(&self, model: &str) -> String {
        self.endpoint_at(&self.url, model)
    }

    /// `endpoint` for a URL held by the caller.
    pub(crate) fn endpoint_at(&self, url: &str, model: &str) -> String {
        let url = url.replace("{{model}}", model);
        if self.custom.is_some() { url } else { self.protocol.endpoint(&url, model) }
    }

    pub(crate) fn headers(&self, key: Option<&str>) -> Vec<(String, String)> {
        let replaces_auth = self.headers.iter().any(|(_, value)| value.contains("{{api_key}}"));
        let mut headers = if replaces_auth || self.custom.is_some() { Vec::new() } else { self.protocol.auth_headers(key) };
        headers.extend(self.headers.iter().map(|(name, value)| (name.clone(), value.replace("{{api_key}}", key.unwrap_or("")))));
        headers
    }

    pub(crate) fn body(&self, request: &LlmRequest) -> Value {
        let Some((template, _)) = &self.custom else { return self.protocol.body(request) };
        // A custom API has no known JSON parameter: the shape goes in the instructions.
        let system = match (&request.schema, request.json) {
            (Some((_, schema)), true) => format!("{}\nReturn only one JSON object matching this JSON Schema exactly, with no other text:\n{schema}", request.system),
            (None, true) => format!("{}\nReturn only one JSON object, with no other text.", request.system),
            _ => request.system.to_string(),
        };
        fill_template(template, request, &system)
    }

    /// `extract_text` with the cut-off flag (custom templates report none).
    pub(crate) fn extract_text_status(&self, payload: &Value) -> Result<(String, bool), String> {
        if self.custom.is_none() {
            return self.protocol.extract_text_status(payload);
        }
        let truncated = self.finish.as_ref().is_some_and(|(pointer, values)| payload.pointer(pointer)
            .map(|reason| reason.as_str().map(str::to_string).unwrap_or_else(|| reason.to_string()))
            .is_some_and(|reason| values.iter().any(|value| value == &reason)));
        self.extract_text(payload).map(|text| (text, truncated))
    }

    pub(crate) fn extract_text(&self, payload: &Value) -> Result<String, String> {
        match &self.custom {
            Some((_, path)) => payload.pointer(path).and_then(Value::as_str).map(str::to_string)
                .filter(|text| !text.is_empty()).ok_or_else(|| "llm_content_missing".into()),
            None => self.protocol.extract_text(payload),
        }
    }

    /// Whether the API has a native JSON parameter that may be requested.
    pub(crate) fn supports_native_json(&self) -> bool {
        if self.command.is_some() {
            return false;
        }
        self.custom.is_none()
    }
}

fn parse_headers(spec: &str) -> Result<Vec<(String, String)>, String> {
    spec.split(';').map(str::trim).filter(|part| !part.is_empty()).map(|part| {
        let (name, value) = part.split_once(':').ok_or_else(|| format!("header \"{part}\" must be Name: value"))?;
        Ok((name.trim().to_string(), value.trim().to_string()))
    }).collect()
}

fn fill_template(template: &Value, request: &LlmRequest, system: &str) -> Value {
    match template {
        Value::String(text) if text == "{{max_tokens}}" => json!(request.max_tokens),
        Value::String(text) if text == "{{temperature}}" => json!(request.temperature),
        Value::String(text) => Value::String(text.replace("{{model}}", request.model)
            .replace("{{system}}", system).replace("{{user}}", &request.user)),
        Value::Array(items) => Value::Array(items.iter().map(|item| fill_template(item, request, system)).collect()),
        Value::Object(map) => Value::Object(map.iter().map(|(key, value)| (key.clone(), fill_template(value, request, system))).collect()),
        other => other.clone(),
    }
}

/// One chat call through the provider egress gateway: the caller's model
/// endpoint in its protocol, workspace scope and approval enforced.
pub(crate) struct LlmCall<'a> {
    pub endpoint: &'a LlmEndpoint,
    pub api_key: Option<&'a str>,
    pub service: &'a str,
    pub purpose: &'a str,
    pub data_classes: &'a [&'a str],
    pub tenant_id: &'a str,
    pub workspace_id: &'a str,
    pub explicit_approval: bool,
}

impl LlmCall<'_> {
    /// Sends the request and returns the reply text.
    /// A server that rejects native JSON output is retried once without it
    /// and remembered, so its later calls skip it.
    pub(crate) async fn complete(&self, request: &LlmRequest<'_>) -> Result<String, String> {
        if self.endpoint.is_command() {
            return self.endpoint.run_command(request.model, &self.endpoint.command_prompt(request)).await;
        }
        let gateway = crate::provider_egress::ProviderEgressGateway::from_env();
        let endpoint = self.endpoint.endpoint(request.model);
        let mut native_json = request.native_json && self.endpoint.supports_native_json() && native_json_allowed(&endpoint);
        loop {
            let body = self.endpoint.body(&LlmRequest { native_json, user: request.user.clone(), schema: request.schema.clone(), ..*request });
            let egress = crate::provider_egress::ProviderEgressRequest::from_env(
                self.service,
                self.purpose,
                self.endpoint.protocol_name(),
                request.model,
                "text",
                self.data_classes,
                body,
            )
            .with_workspace_scope(self.tenant_id, self.workspace_id)
            .with_explicit_user_approval(self.explicit_approval);
            let (response, _) = gateway
                .post_json_with_headers(&endpoint, &self.endpoint.headers(self.api_key), egress)
                .await?;
            let response = gateway.read_json_bounded(response, None).await?;
            if native_json && request.json && rejects_native_json(response.status, &response.payload) {
                remember_no_native_json(&endpoint);
                native_json = false;
                continue;
            }
            if !(200..300).contains(&response.status) {
                return Err(format!("llm_http_status:{}", response.status));
            }
            return self.endpoint.extract_text(&response.payload);
        }
    }
}

/// Splits a command line into argv like a shell would for quoting
/// ("a b", 'a b', a\ b) without running a shell: nothing is expanded or
/// interpreted.
pub(crate) fn split_command(spec: &str) -> Result<Vec<String>, String> {
    let mut argv = Vec::new();
    let mut current = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = spec.chars();
    while let Some(c) = chars.next() {
        match (quote, c) {
            (Some(q), c) if c == q => quote = None,
            (Some('"'), '\\') => current.push(chars.next().ok_or("command ends in an escape")?),
            (Some(_), c) => current.push(c),
            (None, '"' | '\'') => { quote = Some(c); in_word = true; }
            (None, '\\') => { current.push(chars.next().ok_or("command ends in an escape")?); in_word = true; }
            (None, c) if c.is_whitespace() => {
                if in_word { argv.push(std::mem::take(&mut current)); in_word = false; }
            }
            (None, c) => { current.push(c); in_word = true; }
        }
    }
    if quote.is_some() {
        return Err("command has an unclosed quote".into());
    }
    if in_word { argv.push(current); }
    if argv.is_empty() {
        return Err("command is empty".into());
    }
    Ok(argv)
}



/// Tokens one model call used, as the provider reported them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize)]
pub(crate) struct TokenUsage {
    /// All prompt tokens, cached ones included.
    pub input: u64,
    /// Prompt tokens served from the provider's cache (usually billed less).
    pub cached: u64,
    pub output: u64,
}

impl TokenUsage {
    pub(crate) fn add(&mut self, other: TokenUsage) {
        self.input += other.input;
        self.cached += other.cached;
        self.output += other.output;
    }

    /// The usage block of a reply: OpenAI chat (`usage.prompt_tokens`) and
    /// Responses (`usage.input_tokens`), Claude (`usage.input_tokens` plus
    /// cache reads and writes), Gemini (`usageMetadata`). `paths` are JSON
    /// pointers (input, output, cached) for any other API.
    pub(crate) fn from_reply(payload: &Value, paths: &UsagePaths) -> Option<TokenUsage> {
        let number = |pointer: &str| payload.pointer(pointer).and_then(Value::as_u64);
        if let Some(input) = paths.input.as_deref().and_then(number) {
            return Some(TokenUsage {
                input,
                cached: paths.cached.as_deref().and_then(number).unwrap_or(0),
                output: paths.output.as_deref().and_then(number).unwrap_or(0),
            });
        }
        if let Some(meta) = payload.get("usageMetadata") {
            let get = |key: &str| meta.get(key).and_then(Value::as_u64).unwrap_or(0);
            return Some(TokenUsage { input: get("promptTokenCount"), cached: get("cachedContentTokenCount"),
                output: get("candidatesTokenCount") + get("thoughtsTokenCount") });
        }
        let usage = payload.get("usage")?;
        let get = |key: &str| usage.get(key).and_then(Value::as_u64);
        if let Some(prompt) = get("prompt_tokens") {
            let cached = usage.pointer("/prompt_tokens_details/cached_tokens").and_then(Value::as_u64).unwrap_or(0);
            return Some(TokenUsage { input: prompt, cached, output: get("completion_tokens").unwrap_or(0) });
        }
        let input = get("input_tokens")?;
        let (read, write) = (get("cache_read_input_tokens"), get("cache_creation_input_tokens"));
        let cached = read.or_else(|| usage.pointer("/input_tokens_details/cached_tokens").and_then(Value::as_u64)).unwrap_or(0);
        // Claude counts cache reads and writes apart from input_tokens.
        let input = input + read.unwrap_or(0) + write.unwrap_or(0);
        Some(TokenUsage { input, cached, output: get("output_tokens").unwrap_or(0) })
    }

    /// A usage report in a command's output, read with `{PREFIX}_USAGE_REGEX`
    /// (named groups `input`, `output`, `cached`, or `total` counted as input).
    /// Thousands separators are ignored.
    pub(crate) fn from_command_log(log: &str, pattern: &regex::Regex) -> Option<TokenUsage> {
        let caps = pattern.captures_iter(log).last()?;
        let get = |name: &str| caps.name(name).and_then(|m| m.as_str().replace([',', '_', ' '], "").parse::<u64>().ok());
        let input = get("input").or_else(|| get("total"))?;
        Some(TokenUsage { input, cached: get("cached").unwrap_or(0), output: get("output").unwrap_or(0) })
    }
}

/// JSON pointers to a custom API's usage numbers (`{PREFIX}_USAGE_INPUT_PATH`,
/// `_USAGE_OUTPUT_PATH`, `_USAGE_CACHED_PATH`) and a command's usage pattern
/// (`{PREFIX}_USAGE_REGEX`).
#[derive(Debug, Clone, Default)]
pub(crate) struct UsagePaths {
    pub input: Option<String>,
    pub output: Option<String>,
    pub cached: Option<String>,
    pub command: Option<regex::Regex>,
}

impl UsagePaths {
    pub(crate) fn from_env(prefix: &str) -> Self {
        let var = |name: &str| std::env::var(format!("{prefix}_{name}")).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        Self {
            input: var("USAGE_INPUT_PATH"),
            output: var("USAGE_OUTPUT_PATH"),
            cached: var("USAGE_CACHED_PATH"),
            command: var("USAGE_REGEX").and_then(|pattern| regex::Regex::new(&pattern).ok()),
        }
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn token_usage_is_read_from_every_provider_shape() {
        use super::{TokenUsage, UsagePaths};
        let none = UsagePaths::default();
        let usage = |input, cached, output| Some(TokenUsage { input, cached, output });
        assert_eq!(TokenUsage::from_reply(&json!({"usage":{"prompt_tokens":900,"completion_tokens":40,
            "prompt_tokens_details":{"cached_tokens":600}}}), &none), usage(900, 600, 40), "OpenAI chat");
        assert_eq!(TokenUsage::from_reply(&json!({"usage":{"input_tokens":900,"output_tokens":40,
            "input_tokens_details":{"cached_tokens":600}}}), &none), usage(900, 600, 40), "OpenAI responses");
        assert_eq!(TokenUsage::from_reply(&json!({"usage":{"input_tokens":300,"output_tokens":40,
            "cache_read_input_tokens":600,"cache_creation_input_tokens":0}}), &none), usage(900, 600, 40), "Claude counts cache reads apart");
        assert_eq!(TokenUsage::from_reply(&json!({"usageMetadata":{"promptTokenCount":900,"candidatesTokenCount":30,
            "thoughtsTokenCount":10,"cachedContentTokenCount":600}}), &none), usage(900, 600, 40), "Gemini, thinking counted as output");
        assert_eq!(TokenUsage::from_reply(&json!({"choices":[]}), &none), None, "no report is not zero");
        let custom = UsagePaths { input: Some("/stats/in".into()), output: Some("/stats/out".into()), ..UsagePaths::default() };
        assert_eq!(TokenUsage::from_reply(&json!({"stats":{"in":77,"out":5}}), &custom), usage(77, 0, 5), "custom pointers");
        let pattern = regex::Regex::new(r"tokens used:?\s*(?P<total>[\d,]+)").unwrap();
        assert_eq!(TokenUsage::from_command_log("thinking...\ntokens used: 55,771\n", &pattern), usage(55771, 0, 0), "command output");
        assert_eq!(TokenUsage::from_command_log("no report", &pattern), None);
    }

    use super::*;

    pub(super) fn request(strict: bool) -> LlmRequest<'static> {
        LlmRequest {
            model: "m",
            system: "sys",
            user: "question".into(),
            temperature: 0.0,
            json: true,
            schema: Some(("plan", json!({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}))),
            strict,
            native_json: true,
            max_tokens: 4096,
        }
    }

    #[test]
    fn commands_split_like_a_shell_but_nothing_is_interpreted() {
        assert_eq!(split_command(r#"codex exec -m "gpt 6" 'a b' x\ y $HOME;rm"#).unwrap(),
            vec!["codex", "exec", "-m", "gpt 6", "a b", "x y", "$HOME;rm"]);
        assert!(split_command("codex 'open").is_err());
        assert!(split_command("   ").is_err());
    }

    #[test]
    fn a_custom_api_reports_a_cut_off_reply_through_its_configured_finish_reason() {
        let endpoint = LlmEndpoint { protocol: LlmProtocol::OpenAi, url: String::new(), headers: Vec::new(),
            custom: Some((json!({}), "/response".into())), command: None, command_cwd: None,
            finish: Some(("/done_reason".into(), vec!["length".into()])) };
        assert_eq!(endpoint.extract_text_status(&json!({"response":"part","done_reason":"length"})).unwrap(), ("part".into(), true));
        assert_eq!(endpoint.extract_text_status(&json!({"response":"all","done_reason":"stop"})).unwrap(), ("all".into(), false));
        let unconfigured = LlmEndpoint { finish: None, ..endpoint };
        assert_eq!(unconfigured.extract_text_status(&json!({"response":"part","done_reason":"length"})).unwrap(), ("part".into(), false));
    }

    fn command_endpoint(argv: &[&str]) -> LlmEndpoint {
        LlmEndpoint { protocol: LlmProtocol::OpenAi, url: String::new(), headers: Vec::new(), custom: None,
            command: Some(argv.iter().map(|a| a.to_string()).collect()), finish: None, command_cwd: None }
    }

    #[tokio::test]
    async fn a_command_model_runs_in_an_empty_directory_and_is_told_not_to_use_tools() {
        // By default: a fresh empty directory, removed afterwards.
        let endpoint = command_endpoint(&["sh", "-c", "cat > /dev/null; pwd; ls -A | wc -l"]);
        let reply = endpoint.run_command("m", "prompt").await.unwrap();
        let mut lines = reply.lines();
        let dir = lines.next().unwrap().to_string();
        assert!(dir.contains("umem-llm-cwd-"), "{reply}");
        assert_eq!(lines.next().unwrap().trim(), "0", "the directory is empty: {reply}");
        assert!(!std::path::Path::new(&dir).exists(), "removed after the call");
        assert_ne!(std::path::Path::new(&dir), std::env::current_dir().unwrap(), "not the server's directory");
        // An operator-set directory is used as given.
        let chosen = std::env::temp_dir().join(format!("umem-chosen-cwd-{}", std::process::id()));
        std::fs::create_dir_all(&chosen).unwrap();
        let mut endpoint = command_endpoint(&["sh", "-c", "cat > /dev/null; pwd"]);
        endpoint.command_cwd = Some(chosen.to_string_lossy().into_owned());
        let reply = endpoint.run_command("m", "prompt").await.unwrap();
        assert_eq!(std::fs::canonicalize(reply.trim()).unwrap(), std::fs::canonicalize(&chosen).unwrap());
        assert!(chosen.exists(), "an operator's directory is kept");
        let _ = std::fs::remove_dir_all(&chosen);
        // The prompt tells the tool to answer from the input alone.
        let prompt = endpoint.command_prompt(&LlmRequest { model: "m", system: "S", user: "U".into(), temperature: 0.0,
            json: false, schema: None, strict: false, native_json: false, max_tokens: 10 });
        assert!(prompt.starts_with("Answer using only the instructions and input in this message. Do not run commands"), "{prompt}");
    }

    #[tokio::test]
    async fn a_command_model_reads_the_prompt_on_stdin_and_replies_on_stdout_or_a_file() {
        // stdout, with the model filled in
        let stdout = command_endpoint(&["sh", "-c", "cat >/dev/null; printf 'reply from %s' \"$0\"", "{{model}}"]);
        assert_eq!(stdout.run_command("m1", "hello").await.unwrap(), "reply from m1");
        assert_eq!(stdout.protocol_name(), "command");
        assert!(!stdout.supports_native_json());
        // a reply file, as `codex exec --output-last-message` writes
        let file = command_endpoint(&["sh", "-c", "cat > \"$0\"; echo progress-noise", "{{output_file}}"]);
        assert_eq!(file.run_command("m", "the prompt\n").await.unwrap(), "the prompt");
        // a failing tool reports its status and error output
        let failing = command_endpoint(&["sh", "-c", "cat >/dev/null; echo oops >&2; exit 3"]);
        let error = failing.run_command("m", "x").await.unwrap_err();
        assert!(error.starts_with("llm_command_failed") && error.contains("oops"), "{error}");
        // the shared call path sends the instructions with the schema stated
        let echo = command_endpoint(&["cat"]);
        let call = LlmCall { endpoint: &echo, api_key: None, service: "t", purpose: "t", data_classes: &[],
            tenant_id: "t", workspace_id: "w", explicit_approval: true };
        let reply = call.complete(&request(true)).await.unwrap();
        // (after the line telling command tools to answer from the input alone)
        assert!(reply.starts_with("Answer using only the instructions and input in this message.")
            && reply.contains("\n\nsys\nReturn only one JSON object matching this JSON Schema") && reply.ends_with("question"), "{reply}");
    }



    #[test]
    fn json_is_read_from_fences_and_prose() {
        assert_eq!(extract_json("{\"a\":1}").unwrap(), json!({"a":1}));
        assert_eq!(extract_json("```json\n{\"a\":1}\n```").unwrap(), json!({"a":1}));
        assert_eq!(extract_json("Sure! Here it is: {\"a\":{\"b\":\"x } y\"}} Hope this helps.").unwrap(),
            json!({"a":{"b":"x } y"}}));
        assert!(extract_json("I cannot answer that.").is_err());
    }

    #[test]
    fn rejected_json_parameters_are_recognised() {
        assert!(rejects_native_json(400, &json!({"error":{"message":"'response_format.type' must be 'json_schema'"}})));
        assert!(rejects_native_json(422, &json!({"detail":"json_object is not supported"})));
        assert!(!rejects_native_json(400, &json!({"error":{"message":"context length exceeded"}})));
        assert!(!rejects_native_json(500, &json!({"error":"response_format"})));
    }

    #[test]
    fn without_native_json_the_shape_is_requested_in_the_instructions() {
        let mut plain = request(true);
        plain.native_json = false;
        for protocol in [LlmProtocol::OpenAi, LlmProtocol::Anthropic, LlmProtocol::Gemini] {
            let body = protocol.body(&plain);
            assert!(body.get("response_format").is_none() && body.get("output_config").is_none());
            assert!(body.pointer("/generationConfig/responseMimeType").is_none());
            assert!(body.to_string().contains("JSON Schema"), "{protocol:?}");
        }
    }

    #[test]
    fn protocol_is_explicit_or_detected_from_the_url() {
        assert_eq!(LlmProtocol::resolve(None, "https://api.anthropic.com").unwrap(), LlmProtocol::Anthropic);
        assert_eq!(LlmProtocol::resolve(None, "https://generativelanguage.googleapis.com/v1beta").unwrap(), LlmProtocol::Gemini);
        // Gemini's own OpenAI-compatible endpoint speaks the OpenAI protocol.
        assert_eq!(LlmProtocol::resolve(None, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions").unwrap(), LlmProtocol::OpenAi);
        assert_eq!(LlmProtocol::resolve(None, "http://localhost:11434/v1/chat/completions").unwrap(), LlmProtocol::OpenAi);
        assert_eq!(LlmProtocol::resolve(Some("anthropic"), "https://gateway.example/claude").unwrap(), LlmProtocol::Anthropic);
        assert!(LlmProtocol::resolve(Some("soap"), "x").is_err());
    }

    #[test]
    fn endpoints_accept_full_urls_or_bases() {
        assert_eq!(LlmProtocol::Anthropic.endpoint("https://api.anthropic.com", "m"), "https://api.anthropic.com/v1/messages");
        assert_eq!(LlmProtocol::Anthropic.endpoint("https://api.anthropic.com/v1/messages", "m"), "https://api.anthropic.com/v1/messages");
        assert_eq!(LlmProtocol::Gemini.endpoint("https://generativelanguage.googleapis.com/v1beta/", "gemini-x"),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent");
        assert_eq!(LlmProtocol::OpenAi.endpoint("http://h/v1/chat/completions", "m"), "http://h/v1/chat/completions");
        assert_eq!(LlmProtocol::OpenAi.endpoint("https://apihub.example/v1/", "m"), "https://apihub.example/v1/chat/completions");
        assert_eq!(LlmProtocol::OpenAi.endpoint("https://generativelanguage.googleapis.com/v1beta/openai/", "m"),
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions");
        assert_eq!(LlmProtocol::OpenAi.endpoint("https://openrouter.ai/api/v1", "m"), "https://openrouter.ai/api/v1/chat/completions");
        // A custom path is a full endpoint.
        assert_eq!(LlmProtocol::OpenAi.endpoint("http://host:8080/generate", "m"), "http://host:8080/generate");
        assert_eq!(LlmProtocol::OpenAi.endpoint("https://r.openai.azure.com/openai/deployments/d/chat/completions?api-version=1", "m"),
            "https://r.openai.azure.com/openai/deployments/d/chat/completions?api-version=1");
    }

    #[test]
    fn auth_uses_each_protocols_header() {
        assert_eq!(LlmProtocol::OpenAi.auth_headers(Some("k")), vec![("authorization".to_string(), "Bearer k".to_string())]);
        assert!(LlmProtocol::OpenAi.auth_headers(None).is_empty(), "local servers need no key");
        let anthropic = LlmProtocol::Anthropic.auth_headers(Some("k"));
        assert!(anthropic.contains(&("x-api-key".to_string(), "k".to_string())));
        assert!(anthropic.contains(&("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string())));
        assert_eq!(LlmProtocol::Gemini.auth_headers(Some("k")), vec![("x-goog-api-key".to_string(), "k".to_string())]);
    }

    #[test]
    fn bodies_follow_each_api_and_degrade_schemas_to_instructions() {
        let openai = LlmProtocol::OpenAi.body(&request(true));
        assert_eq!(openai["response_format"]["type"], "json_schema");
        assert_eq!(openai["response_format"]["json_schema"]["strict"], true);
        assert_eq!(openai["messages"][0]["content"], "sys");
        let loose = LlmProtocol::OpenAi.body(&request(false));
        assert_eq!(loose["response_format"]["type"], "json_object");
        assert_eq!(loose["messages"][0]["content"], "sys");

        let anthropic = LlmProtocol::Anthropic.body(&request(true));
        assert_eq!(anthropic["system"], "sys");
        assert_eq!(anthropic["max_tokens"], 4096);
        assert_eq!(anthropic["messages"][0]["role"], "user");
        assert_eq!(anthropic["output_config"]["format"]["type"], "json_schema");
        let loose = LlmProtocol::Anthropic.body(&request(false));
        assert!(loose.get("output_config").is_none());
        assert!(loose["system"].as_str().unwrap().contains("JSON Schema"));

        let gemini = LlmProtocol::Gemini.body(&request(true));
        assert_eq!(gemini["generationConfig"]["responseMimeType"], "application/json");
        assert_eq!(gemini["contents"][0]["parts"][0]["text"], "question");
        // Gemini's schema dialect is not full JSON Schema: stated in the prompt.
        assert!(gemini["systemInstruction"]["parts"][0]["text"].as_str().unwrap().contains("JSON Schema"));
    }

    #[test]
    fn reply_text_is_read_per_protocol_and_truncation_is_an_error() {
        assert_eq!(LlmProtocol::OpenAi.extract_text(&json!({"choices":[{"finish_reason":"stop","message":{"content":"{}"}}]})).unwrap(), "{}");
        assert!(LlmProtocol::OpenAi.extract_text(&json!({"choices":[{"finish_reason":"length","message":{"content":"{"}}]})).is_err());
        assert_eq!(LlmProtocol::Anthropic.extract_text(&json!({"stop_reason":"end_turn","content":[{"type":"text","text":"{\"a\":1}"}]})).unwrap(), "{\"a\":1}");
        assert!(LlmProtocol::Anthropic.extract_text(&json!({"stop_reason":"max_tokens","content":[{"type":"text","text":"{"}]})).is_err());
        assert!(LlmProtocol::Anthropic.extract_text(&json!({"stop_reason":"refusal","content":[]})).is_err());
        assert_eq!(LlmProtocol::Gemini.extract_text(&json!({"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"{}"}]}}]})).unwrap(), "{}");
        assert!(LlmProtocol::Gemini.extract_text(&json!({"candidates":[{"finishReason":"SAFETY","content":{"parts":[]}}]})).is_err());
        assert_eq!(LlmProtocol::Gemini.extract_text(&json!({"candidates":[{"finishReason":"STOP","content":{"parts":[
            {"text":"let me think","thought":true},{"text":"{}","thoughtSignature":"abc"}]}}]})).unwrap(), "{}");
    }
}
