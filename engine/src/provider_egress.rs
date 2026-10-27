use chrono::Utc;
use regex::Regex;
use reqwest::{redirect::Policy, Client, Response, Url};
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    env,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, ErrorKind, Write},
    net::IpAddr,
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock, RwLock},
    time::Duration,
};

static RECEIPT_WRITE_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static WORKSPACE_POLICY_PROJECTION: OnceLock<
    RwLock<HashMap<(String, String), ProviderWorkspacePolicyProjection>>,
> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProviderWorkspacePolicyProjection {
    pub(crate) tenant_id: String,
    pub(crate) workspace_id: String,
    pub(crate) mode: String,
    pub(crate) revision: i64,
}

fn workspace_policy_projection(
    tenant_id: &str,
    workspace_id: &str,
) -> Result<ProviderWorkspacePolicyProjection, String> {
    let projection = WORKSPACE_POLICY_PROJECTION
        .get_or_init(|| RwLock::new(HashMap::new()))
        .read()
        .map_err(|_| "provider_workspace_policy_projection_unavailable".to_string())?;
    Ok(projection
        .get(&(tenant_id.to_string(), workspace_id.to_string()))
        .cloned()
        .unwrap_or_else(|| ProviderWorkspacePolicyProjection {
            tenant_id: tenant_id.to_string(),
            workspace_id: workspace_id.to_string(),
            mode: "deployment_policy".to_string(),
            revision: 0,
        }))
}

fn validate_workspace_policy(policy: &ProviderWorkspacePolicyProjection) -> Result<(), String> {
    if policy.tenant_id.trim().is_empty()
        || policy.workspace_id.trim().is_empty()
        || !matches!(
            policy.mode.as_str(),
            "local_only" | "ask_each_time" | "deployment_policy"
        )
        || policy.revision < 0
    {
        return Err("provider_workspace_policy_invalid".to_string());
    }
    Ok(())
}

pub(crate) fn replace_workspace_provider_policies(
    policies: impl IntoIterator<Item = ProviderWorkspacePolicyProjection>,
) -> Result<(), String> {
    let mut next = HashMap::new();
    for policy in policies {
        validate_workspace_policy(&policy)?;
        next.insert(
            (policy.tenant_id.clone(), policy.workspace_id.clone()),
            policy,
        );
    }
    let mut projection = WORKSPACE_POLICY_PROJECTION
        .get_or_init(|| RwLock::new(HashMap::new()))
        .write()
        .map_err(|_| "provider_workspace_policy_projection_unavailable".to_string())?;
    *projection = next;
    Ok(())
}

pub(crate) fn upsert_workspace_provider_policy(
    policy: ProviderWorkspacePolicyProjection,
) -> Result<(), String> {
    validate_workspace_policy(&policy)?;
    let mut projection = WORKSPACE_POLICY_PROJECTION
        .get_or_init(|| RwLock::new(HashMap::new()))
        .write()
        .map_err(|_| "provider_workspace_policy_projection_unavailable".to_string())?;
    projection.insert(
        (policy.tenant_id.clone(), policy.workspace_id.clone()),
        policy,
    );
    Ok(())
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderEgressRequest {
    pub(crate) tenant_id: String,
    pub(crate) workspace_id: String,
    pub(crate) principal_id: String,
    pub(crate) purpose: String,
    pub(crate) provider: String,
    pub(crate) model: String,
    pub(crate) region: String,
    pub(crate) modality: String,
    pub(crate) data_classes: Vec<String>,
    pub(crate) local_only: bool,
    pub(crate) workspace_scope_bound: bool,
    pub(crate) explicit_user_approval: bool,
    pub(crate) payload: Value,
}

impl ProviderEgressRequest {
    pub(crate) fn from_env(
        principal_id: &str,
        purpose: &str,
        provider: &str,
        model: &str,
        modality: &str,
        data_classes: &[&str],
        payload: Value,
    ) -> Self {
        Self {
            tenant_id: env::var("PROVIDER_EGRESS_DEFAULT_TENANT")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "default".to_string()),
            workspace_id: env::var("PROVIDER_EGRESS_DEFAULT_WORKSPACE")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "default".to_string()),
            principal_id: principal_id.to_string(),
            purpose: purpose.to_string(),
            provider: provider.to_string(),
            model: model.to_string(),
            region: env::var("PROVIDER_EGRESS_DEFAULT_REGION")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "unspecified".to_string()),
            modality: modality.to_string(),
            data_classes: data_classes
                .iter()
                .map(|value| (*value).to_string())
                .collect(),
            local_only: env_bool("PROVIDER_EGRESS_LOCAL_ONLY", false),
            workspace_scope_bound: false,
            explicit_user_approval: false,
            payload,
        }
    }

    pub(crate) fn with_workspace_scope(
        mut self,
        tenant_id: impl Into<String>,
        workspace_id: impl Into<String>,
    ) -> Self {
        self.tenant_id = tenant_id.into();
        self.workspace_id = workspace_id.into();
        self.workspace_scope_bound = true;
        self
    }

    pub(crate) fn with_explicit_user_approval(mut self, approved: bool) -> Self {
        self.explicit_user_approval = approved;
        self
    }
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ProviderEgressPreview {
    pub(crate) allowed: bool,
    pub(crate) blocked_reasons: Vec<String>,
    pub(crate) tenant_id: String,
    pub(crate) workspace_id: String,
    pub(crate) principal_id: String,
    pub(crate) purpose: String,
    pub(crate) provider: String,
    pub(crate) model: String,
    pub(crate) region: String,
    pub(crate) modality: String,
    pub(crate) data_classes: Vec<String>,
    pub(crate) destination_scheme: String,
    pub(crate) destination_host: String,
    pub(crate) payload_bytes: usize,
    pub(crate) payload_sha256: String,
    pub(crate) redaction_count: usize,
    pub(crate) policy_id: String,
    pub(crate) workspace_scope_bound: bool,
    pub(crate) workspace_policy_mode: String,
    pub(crate) workspace_policy_revision: i64,
    pub(crate) explicit_user_approval: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderEgressGateway {
    http: Client,
    production: bool,
    allowed_hosts: Arc<HashSet<String>>,
    allowed_data_classes: Arc<HashSet<String>>,
    allow_private_networks: bool,
    max_payload_bytes: usize,
    max_response_bytes: usize,
    receipt_path: Option<PathBuf>,
    policy_id: String,
}

#[derive(Debug, Clone)]
pub(crate) struct ProviderEgressJsonResponse {
    pub(crate) status: u16,
    pub(crate) payload: Value,
    pub(crate) response_bytes: usize,
    pub(crate) response_sha256: String,
}

impl ProviderEgressGateway {
    pub(crate) fn from_env() -> Self {
        let production = env::var("UNLIMITED_MEMORY_PROFILE")
            .ok()
            .map(|value| crate::strict_deployment_profile_name(&value))
            .unwrap_or(false);
        let request_timeout_ms = env::var("PROVIDER_EGRESS_REQUEST_TIMEOUT_MS")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|value| *value >= 100 && *value <= 600_000)
            .unwrap_or(30_000);
        let http = Client::builder()
            .redirect(Policy::none())
            .timeout(Duration::from_millis(request_timeout_ms))
            // reqwest's default TCP keepalive (short idle and probe interval)
            // drops connections whose peer ignores keepalive probes, as some
            // provider front ends do: a model that thinks silently for ~30 s
            // then fails with ETIMEDOUT. The request timeout above already
            // bounds dead connections.
            .tcp_keepalive(None)
            .build()
            .expect("provider egress HTTP client configuration must be valid");
        Self {
            http,
            production,
            allowed_hosts: Arc::new(env_set("PROVIDER_EGRESS_ALLOWED_HOSTS")),
            allowed_data_classes: Arc::new(env_set("PROVIDER_EGRESS_ALLOWED_DATA_CLASSES")),
            allow_private_networks: env_bool("PROVIDER_EGRESS_ALLOW_PRIVATE_NETWORKS", !production),
            max_payload_bytes: env::var("PROVIDER_EGRESS_MAX_PAYLOAD_BYTES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(2 * 1024 * 1024),
            max_response_bytes: env::var("PROVIDER_EGRESS_MAX_RESPONSE_BYTES")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value >= 1024 && *value <= 64 * 1024 * 1024)
                .unwrap_or(4 * 1024 * 1024),
            receipt_path: env::var("PROVIDER_EGRESS_RECEIPT_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|path| !path.as_os_str().is_empty()),
            policy_id: env::var("PROVIDER_EGRESS_POLICY_ID")
                .ok()
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
                .unwrap_or_else(|| "provider-egress-default-v1".to_string()),
        }
    }

    pub(crate) fn preview(
        &self,
        destination: &str,
        request: &ProviderEgressRequest,
    ) -> Result<(ProviderEgressPreview, Value), String> {
        let url = Url::parse(destination)
            .map_err(|error| format!("provider_egress_invalid_url:{error}"))?;
        let host = url
            .host_str()
            .map(str::to_ascii_lowercase)
            .ok_or_else(|| "provider_egress_destination_host_missing".to_string())?;
        let (sanitized_payload, redaction_count) = redact_payload(&request.payload);
        let encoded = serde_json::to_vec(&sanitized_payload)
            .map_err(|error| format!("provider_egress_payload_encode_failed:{error}"))?;
        let mut blocked_reasons = Vec::new();
        let workspace_policy =
            match workspace_policy_projection(&request.tenant_id, &request.workspace_id) {
                Ok(policy) => policy,
                Err(error) => {
                    blocked_reasons.push(error);
                    ProviderWorkspacePolicyProjection {
                        tenant_id: request.tenant_id.clone(),
                        workspace_id: request.workspace_id.clone(),
                        mode: "unavailable".to_string(),
                        revision: -1,
                    }
                }
            };
        match workspace_policy.mode.as_str() {
            "local_only" => {
                blocked_reasons.push("workspace_provider_policy_local_only".to_string())
            }
            "ask_each_time" if !request.explicit_user_approval => {
                blocked_reasons.push("workspace_provider_policy_requires_approval".to_string())
            }
            "deployment_policy" | "ask_each_time" => {}
            _ => blocked_reasons.push("workspace_provider_policy_invalid".to_string()),
        }
        if self.production && !request.workspace_scope_bound {
            blocked_reasons.push("provider_workspace_scope_required".to_string());
        }
        if request.local_only {
            blocked_reasons.push("local_only_policy".to_string());
        }
        if request.tenant_id.trim().is_empty()
            || request.workspace_id.trim().is_empty()
            || request.principal_id.trim().is_empty()
        {
            blocked_reasons.push("missing_tenant_workspace_or_principal".to_string());
        }
        if self.production && url.scheme() != "https" {
            blocked_reasons.push("production_requires_https".to_string());
        }
        if !matches!(url.scheme(), "http" | "https") {
            blocked_reasons.push("unsupported_destination_scheme".to_string());
        }
        if !url.username().is_empty() || url.password().is_some() {
            blocked_reasons.push("destination_userinfo_forbidden".to_string());
        }
        if self.production && self.allowed_hosts.is_empty() {
            blocked_reasons.push("production_host_allowlist_missing".to_string());
        } else if !self.allowed_hosts.is_empty() && !self.allowed_hosts.contains(&host) {
            blocked_reasons.push("destination_host_not_allowed".to_string());
        }
        if !self.allow_private_networks && host_is_private_or_local(&host) {
            blocked_reasons.push("private_or_local_destination_forbidden".to_string());
        }
        if encoded.len() > self.max_payload_bytes {
            blocked_reasons.push("payload_exceeds_policy_limit".to_string());
        }
        for data_class in &request.data_classes {
            if !self.allowed_data_classes.is_empty()
                && !self
                    .allowed_data_classes
                    .contains(&data_class.to_ascii_lowercase())
            {
                blocked_reasons.push(format!("data_class_not_allowed:{data_class}"));
            }
        }
        if self.production && self.allowed_data_classes.is_empty() {
            blocked_reasons.push("production_data_class_allowlist_missing".to_string());
        }
        if self.production && self.receipt_path.is_none() {
            blocked_reasons.push("production_receipt_sink_missing".to_string());
        }

        let preview = ProviderEgressPreview {
            allowed: blocked_reasons.is_empty(),
            blocked_reasons,
            tenant_id: request.tenant_id.clone(),
            workspace_id: request.workspace_id.clone(),
            principal_id: request.principal_id.clone(),
            purpose: request.purpose.clone(),
            provider: request.provider.clone(),
            model: request.model.clone(),
            region: request.region.clone(),
            modality: request.modality.clone(),
            data_classes: request.data_classes.clone(),
            destination_scheme: url.scheme().to_string(),
            destination_host: host,
            payload_bytes: encoded.len(),
            payload_sha256: format!("sha256:{}", hex::encode(Sha256::digest(&encoded))),
            redaction_count,
            policy_id: self.policy_id.clone(),
            workspace_scope_bound: request.workspace_scope_bound,
            workspace_policy_mode: workspace_policy.mode,
            workspace_policy_revision: workspace_policy.revision,
            explicit_user_approval: request.explicit_user_approval,
        };
        Ok((preview, sanitized_payload))
    }

    pub(crate) async fn post_json(
        &self,
        destination: &str,
        bearer_token: Option<&str>,
        request: ProviderEgressRequest,
    ) -> Result<Response, String> {
        self.post_json_with_preview(destination, bearer_token, request)
            .await
            .map(|(response, _)| response)
    }

    pub(crate) async fn post_json_with_preview(
        &self,
        destination: &str,
        bearer_token: Option<&str>,
        request: ProviderEgressRequest,
    ) -> Result<(Response, ProviderEgressPreview), String> {
        let headers = bearer_token
            .filter(|value| !value.is_empty())
            .map(|token| vec![("authorization".to_string(), format!("Bearer {token}"))])
            .unwrap_or_default();
        self.post_json_with_headers(destination, &headers, request).await
    }

    /// Like `post_json_with_preview` with protocol-specific auth headers
    /// (`x-api-key`, `x-goog-api-key`, `anthropic-version`, ...). Headers carry
    /// credentials, so they are never part of the preview or receipts.
    pub(crate) async fn post_json_with_headers(
        &self,
        destination: &str,
        headers: &[(String, String)],
        request: ProviderEgressRequest,
    ) -> Result<(Response, ProviderEgressPreview), String> {
        let (preview, payload) = self.preview(destination, &request)?;
        self.authorize_and_record(&preview)?;
        let mut builder = self.http.post(destination).json(&payload);
        for (name, value) in headers {
            let mut value = reqwest::header::HeaderValue::from_str(value)
                .map_err(|_| "provider_egress_invalid_header_value".to_string())?;
            // Credentials must not appear in debug output, as with bearer_auth.
            let lower = name.to_ascii_lowercase();
            if ["auth", "key", "token", "secret"].iter().any(|marker| lower.contains(marker)) {
                value.set_sensitive(true);
            }
            builder = builder.header(name.as_str(), value);
        }
        let response = builder
            .send()
            .await
            .map_err(|error| format!("provider_egress_http_error:{}", http_error_chain(&error)))?;
        self.record_receipt(&preview, true, "dispatched")?;
        Ok((response, preview))
    }

    pub(crate) async fn read_json_bounded(
        &self,
        mut response: Response,
        requested_max_bytes: Option<usize>,
    ) -> Result<ProviderEgressJsonResponse, String> {
        let max_bytes = requested_max_bytes
            .unwrap_or(self.max_response_bytes)
            .min(self.max_response_bytes)
            .clamp(1024, 64 * 1024 * 1024);
        if response
            .content_length()
            .is_some_and(|length| length > max_bytes as u64)
        {
            return Err("provider_egress_response_exceeds_policy_limit".to_string());
        }
        let status = response.status().as_u16();
        let mut encoded = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| format!("provider_egress_response_read_failed:{error}"))?
        {
            if encoded.len().saturating_add(chunk.len()) > max_bytes {
                return Err("provider_egress_response_exceeds_policy_limit".to_string());
            }
            encoded.extend_from_slice(&chunk);
        }
        let response_sha256 = format!("sha256:{}", hex::encode(Sha256::digest(&encoded)));
        let payload = serde_json::from_slice::<Value>(&encoded)
            .map_err(|error| format!("provider_egress_response_invalid_json:{error}"))?;
        Ok(ProviderEgressJsonResponse {
            status,
            payload,
            response_bytes: encoded.len(),
            response_sha256,
        })
    }

    pub(crate) async fn get_json(
        &self,
        destination: &str,
        request: ProviderEgressRequest,
    ) -> Result<Response, String> {
        let (preview, _) = self.preview(destination, &request)?;
        self.authorize_and_record(&preview)?;
        let response = self
            .http
            .get(destination)
            .send()
            .await
            .map_err(|error| format!("provider_egress_http_error:{}", http_error_chain(&error)))?;
        self.record_receipt(&preview, true, "dispatched")?;
        Ok(response)
    }

    fn authorize_and_record(&self, preview: &ProviderEgressPreview) -> Result<(), String> {
        if !preview.allowed {
            self.record_receipt(preview, false, "blocked")?;
            return Err(format!(
                "provider_egress_blocked:{}",
                preview.blocked_reasons.join(",")
            ));
        }
        self.record_receipt(preview, false, "authorized")
    }

    fn record_receipt(
        &self,
        preview: &ProviderEgressPreview,
        dispatched: bool,
        status: &str,
    ) -> Result<(), String> {
        let Some(path) = self.receipt_path.as_ref() else {
            return if self.production {
                Err("provider_egress_receipt_sink_missing".to_string())
            } else {
                Ok(())
            };
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("provider_egress_receipt_dir_failed:{error}"))?;
        }
        let receipt = serde_json::json!({
            "receipt_format": "provider_egress_disclosure_receipt_v1",
            "created_at": Utc::now(),
            "status": status,
            "dispatched": dispatched,
            "tenant_id": preview.tenant_id,
            "workspace_id": preview.workspace_id,
            "principal_id": preview.principal_id,
            "purpose": preview.purpose,
            "provider": preview.provider,
            "model": preview.model,
            "region": preview.region,
            "modality": preview.modality,
            "data_classes": preview.data_classes,
            "destination_scheme": preview.destination_scheme,
            "destination_host": preview.destination_host,
            "payload_bytes": preview.payload_bytes,
            "payload_sha256": preview.payload_sha256,
            "redaction_count": preview.redaction_count,
            "policy_id": preview.policy_id,
            "workspace_scope_bound": preview.workspace_scope_bound,
            "workspace_policy_mode": preview.workspace_policy_mode,
            "workspace_policy_revision": preview.workspace_policy_revision,
            "explicit_user_approval": preview.explicit_user_approval,
            "blocked_reasons": preview.blocked_reasons
        });
        let line = serde_json::to_string(&receipt)
            .map_err(|error| format!("provider_egress_receipt_encode_failed:{error}"))?;
        let _guard = RECEIPT_WRITE_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| "provider_egress_receipt_lock_poisoned".to_string())?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|error| format!("provider_egress_receipt_open_failed:{error}"))?;
        writeln!(file, "{line}")
            .map_err(|error| format!("provider_egress_receipt_write_failed:{error}"))
    }

    pub(crate) fn load_receipts(&self, limit: usize) -> Result<Vec<Value>, String> {
        let Some(path) = self.receipt_path.as_ref() else {
            return if self.production {
                Err("provider_egress_receipt_sink_missing".to_string())
            } else {
                Ok(Vec::new())
            };
        };
        let _guard = RECEIPT_WRITE_LOCK
            .get_or_init(|| Mutex::new(()))
            .lock()
            .map_err(|_| "provider_egress_receipt_lock_poisoned".to_string())?;
        let file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
            Err(error) => {
                return Err(format!("provider_egress_receipt_open_failed:{error}"));
            }
        };
        let limit = limit.clamp(1, 500);
        let mut receipts = VecDeque::with_capacity(limit);
        for (index, line) in BufReader::new(file).lines().enumerate() {
            let line =
                line.map_err(|error| format!("provider_egress_receipt_read_failed:{error}"))?;
            if line.trim().is_empty() {
                continue;
            }
            let receipt = serde_json::from_str::<Value>(&line).map_err(|error| {
                format!(
                    "provider_egress_receipt_invalid_json:line={}:{}",
                    index + 1,
                    error
                )
            })?;
            if receipt.get("receipt_format").and_then(Value::as_str)
                != Some("provider_egress_disclosure_receipt_v1")
            {
                return Err(format!(
                    "provider_egress_receipt_invalid_format:line={}",
                    index + 1
                ));
            }
            if receipts.len() == limit {
                receipts.pop_front();
            }
            receipts.push_back(receipt);
        }
        Ok(receipts.into_iter().rev().collect())
    }
}

fn env_set(key: &str) -> HashSet<String> {
    env::var(key)
        .ok()
        .into_iter()
        .flat_map(|raw| {
            raw.split(|c| c == ',' || c == ';' || c == '\n' || c == '\r')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_ascii_lowercase)
                .collect::<Vec<_>>()
        })
        .collect()
}

fn env_bool(key: &str, default: bool) -> bool {
    env::var(key)
        .ok()
        .map(|value| match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => true,
            "0" | "false" | "no" | "off" => false,
            _ => default,
        })
        .unwrap_or(default)
}

fn host_is_private_or_local(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") || host.ends_with(".local") {
        return true;
    }
    host.parse::<IpAddr>()
        .map(|ip| match ip {
            IpAddr::V4(ip) => {
                ip.is_private()
                    || ip.is_loopback()
                    || ip.is_link_local()
                    || ip.is_unspecified()
                    || ip.is_broadcast()
            }
            IpAddr::V6(ip) => ip.is_loopback() || ip.is_unspecified() || ip.is_unique_local(),
        })
        .unwrap_or(false)
}

fn redact_payload(value: &Value) -> (Value, usize) {
    let email = Regex::new(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b").unwrap();
    let phone = Regex::new(r"(?x)(?:\+?\d[\d\-\s()]{7,}\d)").unwrap();
    let secret_key =
        Regex::new(r"(?i)(secret|password|api[_-]?key|access[_-]?token|credential)").unwrap();
    fn walk(
        value: &Value,
        email: &Regex,
        phone: &Regex,
        secret_key: &Regex,
        count: &mut usize,
    ) -> Value {
        match value {
            Value::Object(map) => Value::Object(
                map.iter()
                    .map(|(key, value)| {
                        if secret_key.is_match(key) {
                            *count += 1;
                            (key.clone(), Value::String("[REDACTED_SECRET]".to_string()))
                        } else {
                            (key.clone(), walk(value, email, phone, secret_key, count))
                        }
                    })
                    .collect(),
            ),
            Value::Array(items) => Value::Array(
                items
                    .iter()
                    .map(|item| walk(item, email, phone, secret_key, count))
                    .collect(),
            ),
            Value::String(text) => {
                let redacted_email = email.replace_all(text, "[REDACTED_EMAIL]");
                let email_changed = redacted_email.as_ref() != text.as_str();
                let (redacted, phone_changed) =
                    if chrono::DateTime::parse_from_rfc3339(text).is_ok() {
                        (redacted_email.into_owned(), false)
                    } else {
                        let phone_redacted = phone.replace_all(&redacted_email, "[REDACTED_PHONE]");
                        let phone_changed = phone_redacted != redacted_email;
                        (phone_redacted.into_owned(), phone_changed)
                    };
                if email_changed {
                    *count += 1;
                }
                if phone_changed {
                    *count += 1;
                }
                Value::String(redacted)
            }
            _ => value.clone(),
        }
    }
    let mut count = 0;
    let redacted = walk(value, &email, &phone, &secret_key, &mut count);
    (redacted, count)
}



/// reqwest's Display hides the cause ("error sending request"); timeouts and
/// connection resets differ only in the source chain.
fn http_error_chain(error: &reqwest::Error) -> String {
    let mut text = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(cause) = source {
        text.push_str(": ");
        text.push_str(&cause.to_string());
        source = cause.source();
    }
    if error.is_timeout() {
        text.push_str(" (timeout)");
    }
    text
}


#[cfg(test)]
mod tests {
    use super::*;

    fn test_gateway(receipt_path: PathBuf) -> ProviderEgressGateway {
        ProviderEgressGateway {
            http: Client::new(),
            production: false,
            allowed_hosts: Arc::new(HashSet::from([
                "api.example.com".to_string(),
                "127.0.0.1".to_string(),
            ])),
            allowed_data_classes: Arc::new(HashSet::from(["text".to_string()])),
            allow_private_networks: false,
            max_payload_bytes: 64 * 1024,
            max_response_bytes: 64 * 1024,
            receipt_path: Some(receipt_path),
            policy_id: "test-egress-v1".to_string(),
        }
    }

    #[test]
    fn preview_should_redact_dlp_values_without_returning_raw_secrets() {
        let path = PathBuf::from(format!(
            "/tmp/provider-egress-preview-{}.jsonl",
            std::process::id()
        ));
        let gateway = test_gateway(path);
        let request = ProviderEgressRequest {
            tenant_id: "tenant-a".to_string(),
            workspace_id: "workspace-a".to_string(),
            principal_id: "service:test".to_string(),
            purpose: "test_preview".to_string(),
            provider: "test".to_string(),
            model: "test-model".to_string(),
            region: "test-region".to_string(),
            modality: "text".to_string(),
            data_classes: vec!["text".to_string()],
            local_only: false,
            workspace_scope_bound: true,
            explicit_user_approval: false,
            payload: serde_json::json!({
                "message": "contact alice@example.com or +1 212 555 0199",
                "api_key": "must-never-leave"
            }),
        };

        let (preview, sanitized) = gateway
            .preview("https://api.example.com/v1/chat", &request)
            .unwrap();
        assert!(preview.allowed);
        assert!(preview.redaction_count >= 3);
        let rendered = sanitized.to_string();
        assert!(!rendered.contains("alice@example.com"));
        assert!(!rendered.contains("must-never-leave"));
        assert!(rendered.contains("[REDACTED_EMAIL]"));
        assert!(rendered.contains("[REDACTED_SECRET]"));
    }

    #[test]
    fn preview_should_preserve_rfc3339_timestamps() {
        let path = PathBuf::from(format!(
            "/tmp/provider-egress-timestamp-{}.jsonl",
            std::process::id()
        ));
        let gateway = test_gateway(path);
        let timestamp = "2026-08-09T12:00:00Z";
        let request = ProviderEgressRequest {
            tenant_id: "tenant-a".to_string(),
            workspace_id: "workspace-a".to_string(),
            principal_id: "service:test".to_string(),
            purpose: "timestamp_preview".to_string(),
            provider: "test".to_string(),
            model: "test-model".to_string(),
            region: "test-region".to_string(),
            modality: "text".to_string(),
            data_classes: vec!["text".to_string()],
            local_only: false,
            workspace_scope_bound: true,
            explicit_user_approval: false,
            payload: serde_json::json!({"predicted_for": timestamp}),
        };

        let (preview, sanitized) = gateway
            .preview("https://api.example.com/v1/predict", &request)
            .unwrap();
        assert!(preview.allowed);
        assert_eq!(preview.redaction_count, 0);
        assert_eq!(sanitized["predicted_for"], timestamp);
    }

    #[tokio::test]
    async fn blocked_request_should_write_receipt_without_network_dispatch() {
        let path = PathBuf::from(format!(
            "/tmp/provider-egress-blocked-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let gateway = test_gateway(path.clone());
        let request = ProviderEgressRequest {
            tenant_id: "tenant-a".to_string(),
            workspace_id: "workspace-a".to_string(),
            principal_id: "service:test".to_string(),
            purpose: "blocked_test".to_string(),
            provider: "test".to_string(),
            model: "test-model".to_string(),
            region: "test-region".to_string(),
            modality: "telemetry".to_string(),
            data_classes: vec!["telemetry".to_string()],
            local_only: false,
            workspace_scope_bound: true,
            explicit_user_approval: false,
            payload: serde_json::json!({"sample": 1}),
        };

        let error = gateway
            .post_json("http://127.0.0.1:9/v1", None, request)
            .await
            .expect_err("policy must block before dispatch");
        assert!(error.starts_with("provider_egress_blocked:"));
        let receipt = fs::read_to_string(path).unwrap();
        assert!(receipt.contains("\"status\":\"blocked\""));
        assert!(receipt.contains("\"dispatched\":false"));
        assert!(!receipt.contains("\"sample\":1"));
    }

    #[test]
    fn receipt_reader_should_return_newest_bounded_redacted_records() {
        let path = PathBuf::from(format!(
            "/tmp/provider-egress-list-{}.jsonl",
            std::process::id()
        ));
        let _ = fs::remove_file(&path);
        let gateway = test_gateway(path);
        let request = ProviderEgressRequest {
            tenant_id: "tenant-a".to_string(),
            workspace_id: "workspace-a".to_string(),
            principal_id: "service:test".to_string(),
            purpose: "list_test".to_string(),
            provider: "test".to_string(),
            model: "test-model".to_string(),
            region: "test-region".to_string(),
            modality: "text".to_string(),
            data_classes: vec!["text".to_string()],
            local_only: false,
            workspace_scope_bound: true,
            explicit_user_approval: false,
            payload: serde_json::json!({"secret": "must-never-appear"}),
        };
        let (preview, _) = gateway
            .preview("https://api.example.com/v1/chat", &request)
            .unwrap();
        gateway
            .record_receipt(&preview, false, "authorized")
            .unwrap();
        gateway
            .record_receipt(&preview, true, "dispatched")
            .unwrap();

        let receipts = gateway.load_receipts(1).unwrap();
        assert_eq!(receipts.len(), 1);
        assert_eq!(receipts[0]["status"], "dispatched");
        assert_eq!(receipts[0]["workspace_id"], "workspace-a");
        assert!(!receipts[0].to_string().contains("must-never-appear"));
    }

    #[test]
    fn workspace_policy_should_block_local_only_and_require_explicit_approval() {
        let path = PathBuf::from(format!(
            "/tmp/provider-egress-workspace-policy-{}.jsonl",
            std::process::id()
        ));
        let gateway = test_gateway(path);
        let mut request = ProviderEgressRequest {
            tenant_id: "tenant-policy-test".to_string(),
            workspace_id: "workspace-policy-test".to_string(),
            principal_id: "human:test".to_string(),
            purpose: "policy_test".to_string(),
            provider: "test".to_string(),
            model: "test-model".to_string(),
            region: "test-region".to_string(),
            modality: "text".to_string(),
            data_classes: vec!["text".to_string()],
            local_only: false,
            workspace_scope_bound: true,
            explicit_user_approval: false,
            payload: serde_json::json!({"message": "safe"}),
        };
        upsert_workspace_provider_policy(ProviderWorkspacePolicyProjection {
            tenant_id: request.tenant_id.clone(),
            workspace_id: request.workspace_id.clone(),
            mode: "local_only".to_string(),
            revision: 2,
        })
        .unwrap();
        let (preview, _) = gateway
            .preview("https://api.example.com/v1/chat", &request)
            .unwrap();
        assert!(!preview.allowed);
        assert!(preview
            .blocked_reasons
            .contains(&"workspace_provider_policy_local_only".to_string()));

        upsert_workspace_provider_policy(ProviderWorkspacePolicyProjection {
            tenant_id: request.tenant_id.clone(),
            workspace_id: request.workspace_id.clone(),
            mode: "ask_each_time".to_string(),
            revision: 3,
        })
        .unwrap();
        let (preview, _) = gateway
            .preview("https://api.example.com/v1/chat", &request)
            .unwrap();
        assert!(!preview.allowed);
        assert!(preview
            .blocked_reasons
            .contains(&"workspace_provider_policy_requires_approval".to_string()));
        request.explicit_user_approval = true;
        let (preview, _) = gateway
            .preview("https://api.example.com/v1/chat", &request)
            .unwrap();
        assert!(preview.allowed);
        assert_eq!(preview.workspace_policy_mode, "ask_each_time");
        assert_eq!(preview.workspace_policy_revision, 3);
    }
}
