//! Tidemark: a memory layer with two operations.
//!
//! - `POST /add`: store messages verbatim, one record per message (long
//!   messages are split), each with its date and notes resolving relative
//!   time ("yesterday = Sunday 7 May 2023"). Returns after the records are
//!   durable and searchable.
//! - `POST /search`: rank one user's records by BM25 and embedding
//!   similarity (reciprocal-rank fusion) and return at most `top_k`.
//!
//! Search returns stored records only; it never writes answers.

mod dates;
mod embed;
mod rerank;
mod search;

use axum::{extract::State, http::{HeaderMap, StatusCode}, response::IntoResponse, routing::{get, post}, Json, Router};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
struct Record {
    id: String,
    /// The speaker named at the start of the message ("Caroline: …"), if any.
    speaker: Option<String>,
    content: String,
    created_at: Option<String>,
    tokens: Vec<String>,
    embedding: Vec<f32>,
    /// Tokens of the user's own lines (empty when the record has none).
    user_tokens: Vec<String>,
    /// The user's lines that state a standing instruction or preference.
    instructions: Vec<String>,
}

/// The user-line tokens and standing instructions of a record's content.
fn user_parts(content: &str) -> (Vec<String>, Vec<String>) {
    let lines = search::user_lines(content);
    let tokens = lines.iter().flat_map(|l| search::tokens(l)).collect();
    let instructions = lines.into_iter().filter(|l| search::is_standing_instruction(l)).map(|l| l.chars().take(1000).collect()).collect();
    (tokens, instructions)
}

struct AppState {
    db: Mutex<Connection>,
    /// Users held in memory (records, BM25 tokens, embeddings, engine), least
    /// recently used evicted beyond `cache_budget` bytes. SQLite holds everything.
    cache: Mutex<HashMap<String, CachedUser>>,
    cache_budget: usize,
    use_engine: bool,
    clock: std::sync::atomic::AtomicU64,
    /// Serializes Adds of one user so retried request IDs stay idempotent;
    /// different users are added in parallel.
    add_locks: Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
    /// Bounds concurrent requests; beyond it requests get 429 + Retry-After.
    permits: Arc<tokio::sync::Semaphore>,
    embedder: embed::Embedder,
    reranker: Option<Arc<rerank::Reranker>>,
    keys: Vec<String>,
    settings: Settings,
}

struct CachedUser {
    user: Arc<Mutex<UserState>>,
    /// The user's estimated size, readable without locking the user.
    bytes: Arc<std::sync::atomic::AtomicUsize>,
    last_used: u64,
}

/// One user's records and, with `TIDEMARK_BACKEND=engine`, their own memory engine.
struct UserState {
    records: Vec<Record>,
    engine: Option<EngineBackend>,
    bytes: Arc<std::sync::atomic::AtomicUsize>,
}

impl UserState {
    fn push(&mut self, user_id: &str, record: Record) {
        if let Some(engine) = &mut self.engine {
            engine.store(user_id, &record);
        }
        self.bytes.fetch_add(record_bytes(&record), std::sync::atomic::Ordering::Relaxed);
        self.records.push(record);
    }
}

/// Estimated memory use of a record in the service and the engine (text kept
/// in several forms plus derived memories and tokens, embedding). Calibrated
/// on LongMemEval-S: about 36 bytes per byte of record text.
fn record_bytes(record: &Record) -> usize {
    record.content.len() * 36 + record.embedding.len() * 8 + 2048
}

/// The memory engine plus the mapping from its memory IDs to Tidemark records.
struct EngineBackend {
    engine: tidemark_engine::Engine,
    records: HashMap<u64, (String, Option<String>)>,
    rejected: usize,
}

impl EngineBackend {
    fn store(&mut self, user_id: &str, record: &Record) {
        let stored = self.engine.ingest(user_id, &record.content, "tidemark");
        match stored.memory_id {
            Some(id) if stored.accepted => {
                self.records.insert(id, (record.id.clone(), record.created_at.clone()));
            }
            _ => self.rejected += 1,
        }
    }
}

#[derive(Clone)]
struct Settings {
    /// Words per record before a long message is split.
    max_words: usize,
    /// Characters of the previous message shown as context (0 = off).
    previous_context_chars: usize,
    /// Weight of the lexical ranking in the fusion (0..1).
    lexical_weight: f32,
    rrf_k: f32,
    /// Extra weight for records spoken by a person the query names (0 = off).
    speaker_boost: f32,
    /// Consecutive messages of one Add request per record (1 = one message
    /// per record). Blocks do not overlap.
    window: usize,
    /// Fused candidates re-ordered by the cross-encoder (0 = off).
    rerank_candidates: usize,
    /// What the cross-encoder re-orders: the hybrid candidates only, with the
    /// engine's results kept in front ("hybrid"), or the combined list of
    /// engine and hybrid candidates ("all"), so weak engine hits can drop.
    rerank_all: bool,
    /// Most results returned even when top_k asks for more (0 = top_k).
    max_results: usize,
    /// Most characters of content returned in total; the first result is
    /// always returned (0 = no limit).
    max_chars: usize,
    /// Weight of the ranking over the user's own lines, added to the main
    /// ranking (0 = off).
    user_weight: f32,
    /// Slots at the end of the results for the user's standing instructions
    /// and preferences, most relevant first (0 = off).
    instruction_slots: usize,
    /// Every part of a split named message repeats the speaker
    /// (TIDEMARK_PART_SPEAKERS=on; off = as evaluated).
    part_speakers: bool,
}

impl Settings {
    fn from_env() -> Self {
        let num = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.trim().parse::<f32>().ok()).unwrap_or(d);
        Settings {
            max_words: num("TIDEMARK_MAX_WORDS", 300.0) as usize,
            previous_context_chars: num("TIDEMARK_PREVIOUS_CONTEXT_CHARS", 160.0) as usize,
            lexical_weight: num("TIDEMARK_LEXICAL_WEIGHT", 0.5).clamp(0.0, 1.0),
            rrf_k: num("TIDEMARK_RRF_K", 60.0),
            speaker_boost: num("TIDEMARK_SPEAKER_BOOST", 0.0).max(0.0),
            window: (num("TIDEMARK_WINDOW", 5.0) as usize).max(1),
            rerank_candidates: num("TIDEMARK_RERANK_CANDIDATES", 0.0) as usize,
            rerank_all: std::env::var("TIDEMARK_RERANK_SCOPE").is_ok_and(|v| v.trim().eq_ignore_ascii_case("all")),
            max_results: num("TIDEMARK_MAX_RESULTS", 0.0) as usize,
            max_chars: num("TIDEMARK_MAX_CHARS", 0.0) as usize,
            user_weight: num("TIDEMARK_USER_WEIGHT", 1.0).max(0.0),
            instruction_slots: num("TIDEMARK_INSTRUCTION_SLOTS", 10.0) as usize,
            part_speakers: std::env::var("TIDEMARK_PART_SPEAKERS").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")),
        }
    }
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    role: String,
    #[serde(default)]
    timestamp: Option<i64>,
    content: serde_json::Value,
}

#[derive(Deserialize)]
struct AddRequest {
    request_id: String,
    messages: Vec<Message>,
    user_id: String,
    session_id: String,
}

#[derive(Deserialize)]
struct SearchRequest {
    query: serde_json::Value,
    #[serde(default)]
    options: Vec<serde_json::Value>,
    user_id: String,
    #[serde(default = "default_top_k")]
    top_k: usize,
    /// Per-request overrides of TIDEMARK_MAX_RESULTS and TIDEMARK_MAX_CHARS.
    #[serde(default)]
    max_results: Option<usize>,
    #[serde(default)]
    max_chars: Option<usize>,
    /// Per-request overrides of TIDEMARK_USER_WEIGHT and TIDEMARK_INSTRUCTION_SLOTS.
    #[serde(default)]
    user_weight: Option<f32>,

    #[serde(default)]
    instruction_slots: Option<usize>,
    /// Per-request overrides of TIDEMARK_RERANK_CANDIDATES and TIDEMARK_RERANK_SCOPE.
    #[serde(default)]
    rerank_candidates: Option<usize>,
    #[serde(default)]
    rerank_scope: Option<String>,
}

fn default_top_k() -> usize {
    100
}

fn error(status: StatusCode, message: impl Into<String>) -> axum::response::Response {
    (status, Json(serde_json::json!({"success": false, "error": message.into()}))).into_response()
}

/// A request permit, or 429 with Retry-After when the service is at capacity.
fn permit(state: &AppState) -> Result<tokio::sync::OwnedSemaphorePermit, axum::response::Response> {
    state.permits.clone().try_acquire_owned().map_err(|_| {
        let mut response = error(StatusCode::TOO_MANY_REQUESTS, "busy; retry after the indicated delay");
        response.headers_mut().insert(axum::http::header::RETRY_AFTER, axum::http::HeaderValue::from_static("2"));
        response
    })
}

fn authorized(state: &AppState, headers: &HeaderMap) -> bool {
    if state.keys.is_empty() {
        return true;
    }
    let header = |name: &str| headers.get(name).and_then(|v| v.to_str().ok()).map(str::trim);
    let presented = header("x-api-key").map(str::to_string).or_else(|| {
        header("authorization").and_then(|v| {
            let mut parts = v.splitn(2, ' ');
            let scheme = parts.next()?.to_ascii_lowercase();
            let token = parts.next()?.trim();
            (scheme == "bearer" || scheme == "token").then(|| token.to_string())
        })
    });
    // Constant-time comparison: response timing reveals nothing about the key.
    presented.is_some_and(|k| {
        use subtle::ConstantTimeEq;
        state.keys.iter().fold(false, |found, key| found | bool::from(key.as_bytes().ct_eq(k.as_bytes())))
    })
}

/// Text of a message's content: a string, or the text parts of a part list.
fn content_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        other => other.to_string(),
    }
}

/// Characters per record part; text without spaces (e.g. Chinese) is split by
/// characters, since word counts do not apply to it.
const MAX_PART_CHARS: usize = 2000;

/// True when `text` must be split: more than `max_words` words or more than
/// `MAX_PART_CHARS` characters.
fn too_long(text: &str, max_words: usize) -> bool {
    (max_words > 0 && text.split_whitespace().count() > max_words) || text.chars().count() > MAX_PART_CHARS
}

/// Splits `text` into overlapping parts of at most `max_words` words and
/// `MAX_PART_CHARS` characters (words longer than 400 characters are cut).
fn split_words(text: &str, max_words: usize) -> Vec<String> {
    if !too_long(text, max_words) {
        return vec![text.trim().to_string()];
    }
    let mut pieces: Vec<String> = Vec::new();
    for word in text.split_whitespace() {
        let chars = word.chars().collect::<Vec<_>>();
        for chunk in chars.chunks(400) {
            pieces.push(chunk.iter().collect());
        }
    }
    let max_words = if max_words == 0 { usize::MAX } else { max_words };
    let mut out = Vec::new();
    let mut start = 0;
    while start < pieces.len() {
        let (mut end, mut chars) = (start, 0usize);
        while end < pieces.len() && end - start < max_words && (end == start || chars + pieces[end].chars().count() + 1 <= MAX_PART_CHARS) {
            chars += pieces[end].chars().count() + 1;
            end += 1;
        }
        out.push(pieces[start..end].join(" "));
        if end >= pieces.len() {
            break;
        }
        // 20% overlap, always moving forward
        start = (start + ((end - start) * 4 / 5).max(1)).min(end);
    }
    out
}

fn date_header(ts: Option<i64>) -> (String, Option<chrono::NaiveDate>, Option<String>) {
    match ts.and_then(DateTime::<Utc>::from_timestamp_millis) {
        Some(dt) => (
            format!("[{}] ", dt.format("%A %-d %B %Y, %H:%M")),
            Some(dt.date_naive()),
            Some(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        ),
        None => (String::new(), None, None),
    }
}

/// The records of one message, as the reader will see them.
fn render(role: &str, text: &str, ts: Option<i64>, previous: Option<&str>, s: &Settings) -> Vec<(String, Option<String>)> {
    let (header, date, created_at) = date_header(ts);
    // A message that already names its speaker ("Caroline: …") keeps it.
    let named = speaker_of(text).is_some();
    let speaker = if named || role.trim().is_empty() { String::new() } else { format!("{}: ", role.trim()) };
    let context = match previous {
        Some(p) if s.previous_context_chars > 0 && !p.trim().is_empty() => {
            let p = p.trim();
            let cut = if p.chars().count() > s.previous_context_chars {
                format!("{}…", p.chars().take(s.previous_context_chars).collect::<String>())
            } else {
                p.to_string()
            };
            format!(" (in reply to: {cut})")
        }
        _ => String::new(),
    };
    // A message that names its speaker has the name in its first part only;
    // later parts repeat it.
    let name = if named { text.split_once(':').map(|(head, _)| format!("{}: ", head.trim())).unwrap_or_default() } else { String::new() };
    split_words(text, s.max_words)
        .into_iter()
        .enumerate()
        .map(|(pi, part)| {
            let notes = date.map(|d| dates::annotate(&part, d)).unwrap_or_default();
            let notes = if notes.is_empty() { String::new() } else { format!(" [time: {}]", notes.join("; ")) };
            let speaker = if pi > 0 && named && s.part_speakers && !part.starts_with(name.as_str()) { name.as_str() } else { speaker.as_str() };
            (format!("{header}{speaker}{part}{notes}{context}"), created_at.clone())
        })
        .collect()
}

/// The text embedded for each record of one message: the message itself when
/// it fits in one record; otherwise each part's own words, led by the speaker
/// ("Caroline: …" or the role), so parts get their own vectors and no text is
/// embedded twice.
fn embed_texts(role: &str, text: &str, s: &Settings) -> Vec<String> {
    let parts = split_words(text, s.max_words);
    if parts.len() == 1 {
        return vec![text.to_string()];
    }
    let speaker = match speaker_of(text) {
        Some(_) => text.split_once(':').map(|(head, _)| format!("{}: ", head.trim())).unwrap_or_default(),
        None if !role.trim().is_empty() => format!("{}: ", role.trim()),
        None => String::new(),
    };
    parts.into_iter().map(|part| if part.starts_with(speaker.as_str()) { part } else { format!("{speaker}{part}") }).collect()
}

/// One user's records as stored in SQLite (read under the database lock).
fn read_user(db: &Connection, user_id: &str) -> rusqlite::Result<Vec<Record>> {
    let mut stmt = db.prepare("SELECT id, content, created_at, embedding, raw FROM records WHERE user_id = ?1 ORDER BY seq")?;
    let rows = stmt.query_map(params![user_id], |r| {
        let blob: Vec<u8> = r.get(3)?;
        let content: String = r.get(1)?;
        let (user_tokens, instructions) = user_parts(&content);
        Ok(Record {
            user_tokens,
            instructions,
            id: r.get(0)?,
            speaker: speaker_of(&r.get::<_, String>(4)?),
            tokens: search::tokens(&content),
            content,
            created_at: r.get(2)?,
            embedding: blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        })
    })?;
    rows.collect()
}

/// A user's in-memory state built from their records (engine replay is CPU
/// work and runs without the database lock).
fn build_user(user_id: &str, records: Vec<Record>, use_engine: bool) -> UserState {
    let mut user = UserState {
        records: Vec::with_capacity(records.len()),
        engine: use_engine.then(|| EngineBackend { engine: tidemark_engine::Engine::new(), records: HashMap::new(), rejected: 0 }),
        bytes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
    };
    for record in records {
        user.push(user_id, record);
    }
    user
}

/// The cached state of `user_id`, loading it from SQLite when absent. The
/// caller must hold the user's lock (`add_locks`), so a user is loaded once.
async fn user_state(state: &Arc<AppState>, user_id: &str) -> Result<Arc<Mutex<UserState>>, String> {
    let tick = state.clock.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if let Some(cached) = state.cache.lock().unwrap().get_mut(user_id) {
        cached.last_used = tick;
        return Ok(cached.user.clone());
    }
    let st = state.clone();
    let id = user_id.to_string();
    let loaded = tokio::task::spawn_blocking(move || -> Result<UserState, String> {
        let records = {
            let db = st.db.lock().unwrap();
            read_user(&db, &id).map_err(|e| e.to_string())?
        };
        Ok(build_user(&id, records, st.use_engine))
    })
    .await
    .map_err(|e| e.to_string())??;
    let bytes = loaded.bytes.clone();
    let user = Arc::new(Mutex::new(loaded));
    let mut cache = state.cache.lock().unwrap();
    cache.insert(user_id.to_string(), CachedUser { user: user.clone(), bytes, last_used: tick });
    evict(&mut cache, state.cache_budget);
    Ok(user)
}

/// Drops least recently used users until the estimated size fits the budget;
/// users in use elsewhere (their state shared) are kept.
fn evict(cache: &mut HashMap<String, CachedUser>, budget: usize) {
    let size_of = |u: &CachedUser| u.bytes.load(std::sync::atomic::Ordering::Relaxed);
    let mut total = cache.values().map(size_of).sum::<usize>();
    while total > budget {
        let victim = cache
            .iter()
            .filter(|(_, u)| Arc::strong_count(&u.user) == 1)
            .min_by_key(|(_, u)| u.last_used)
            .map(|(k, u)| (k.clone(), size_of(u)));
        match victim {
            Some((key, bytes)) if cache.len() > 1 => {
                cache.remove(&key);
                total = total.saturating_sub(bytes);
                if std::env::var_os("TIDEMARK_LOG_CACHE").is_some() {
                    eprintln!("cache: evicted {key} ({} MB); {} users, {} MB cached", bytes >> 20, cache.len(), total >> 20);
                }
            }
            _ => break,
        }
    }
}

/// The lock serializing Adds and loading of one user.
fn user_lock(state: &AppState, user_id: &str) -> Arc<tokio::sync::Mutex<()>> {
    state.add_locks.lock().unwrap().entry(user_id.to_string()).or_default().clone()
}

/// One line of a block: the message with its speaker and time notes.
fn block_line(role: &str, text: &str, ts: Option<i64>) -> String {
    let (_, date, _) = date_header(ts);
    let speaker = if speaker_of(text).is_some() || role.trim().is_empty() { String::new() } else { format!("{}: ", role.trim()) };
    let notes = date.map(|d| dates::annotate(text, d)).unwrap_or_default();
    let notes = if notes.is_empty() { String::new() } else { format!(" [time: {}]", notes.join("; ")) };
    format!("{speaker}{}{notes}", text.trim())
}

/// Records for consecutive messages grouped `window` at a time: one date
/// header (the first message's) and one line per message. A message longer
/// than `max_words` is left out of blocks and split on its own.
fn render_blocks(messages: &[(String, String, Option<i64>)], s: &Settings) -> Vec<(String, String, Option<String>)> {
    let mut out = Vec::new();
    let mut block: Vec<&(String, String, Option<i64>)> = Vec::new();
    let flush = |block: &mut Vec<&(String, String, Option<i64>)>, out: &mut Vec<(String, String, Option<String>)>| {
        if block.is_empty() {
            return;
        }
        let (header, _, created_at) = date_header(block[0].2);
        let lines = block.iter().map(|(role, text, ts)| block_line(role, text, *ts)).collect::<Vec<_>>();
        let raw = block.iter().map(|(_, text, _)| text.trim()).collect::<Vec<_>>().join("\n");
        out.push((raw, format!("{header}\n{}", lines.join("\n")).trim_start().to_string(), created_at));
        block.clear();
    };
    for m in messages {
        if too_long(&m.1, s.max_words) {
            flush(&mut block, &mut out);
            for ((content, created_at), part) in render(&m.0, &m.1, m.2, None, s).into_iter().zip(embed_texts(&m.0, &m.1, s)) {
                out.push((part, content, created_at));
            }
            continue;
        }
        // A block shows one date, so a message from another day starts a new
        // block (messages of the same day keep sharing one).
        if block.first().is_some_and(|first| date_header(first.2).1 != date_header(m.2).1) {
            flush(&mut block, &mut out);
        }
        block.push(m);
        if block.len() == s.window {
            flush(&mut block, &mut out);
        }
    }
    flush(&mut block, &mut out);
    out
}

async fn add(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Result<Json<AddRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let _permit = match permit(&state) {
        Ok(p) => p,
        Err(busy) => return busy,
    };
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let ok = || Json(serde_json::json!({"success": true, "request_id": req.request_id, "user_id": req.user_id, "session_id": req.session_id})).into_response();
    let lock = user_lock(&state, &req.user_id);
    let _guard = lock.lock().await;
    let seen = {
        let db = state.db.lock().unwrap();
        db.query_row("SELECT 1 FROM user_requests WHERE user_id = ?1 AND request_id = ?2", params![req.user_id, req.request_id], |_| Ok(())).is_ok()
    };
    if seen {
        return ok();
    }
    // The previous message for context: the last one stored in this session.
    let mut previous: Option<String> = {
        let db = state.db.lock().unwrap();
        db.query_row(
            "SELECT raw FROM records WHERE user_id = ?1 AND session_id = ?2 ORDER BY seq DESC LIMIT 1",
            params![req.user_id, req.session_id],
            |r| r.get::<_, String>(0),
        )
        .ok()
    };
    // Requests are deduplicated per user, and record IDs are namespaced the
    // same way, so two users reusing a request ID neither lose writes nor collide.
    let request_key = short_hash(&format!("{}\u{0}{}", req.user_id, req.request_id));
    let mut rows = Vec::new();
    if state.settings.window > 1 {
        let messages = req.messages.iter().map(|m| (m.role.clone(), content_text(&m.content), m.timestamp))
            .filter(|m| !m.1.trim().is_empty()).collect::<Vec<_>>();
        for (bi, (raw, content, created_at)) in render_blocks(&messages, &state.settings).into_iter().enumerate() {
            rows.push((format!("{}#b{}", request_key, bi), raw.clone(), raw, content, created_at));
        }
    }
    for (mi, m) in req.messages.iter().enumerate().filter(|_| state.settings.window == 1) {
        let text = content_text(&m.content);
        if text.trim().is_empty() {
            continue;
        }
        let parts = embed_texts(&m.role, &text, &state.settings);
        for (pi, ((content, created_at), part)) in render(&m.role, &text, m.timestamp, previous.as_deref(), &state.settings).into_iter().zip(parts).enumerate() {
            let id = format!("{}#{}.{}", request_key, mi, pi);
            // The whole message stays the stored text (the next message's context).
            rows.push((id, text.clone(), part, content, created_at));
        }
        previous = Some(text);
    }
    // Embed the speaker and words; dates are matched lexically.
    let embed_inputs = rows.iter().map(|(_, _, embed, _, _)| embed.clone()).collect::<Vec<_>>();
    let vectors = match state.embedder.embed_counted(&embed_inputs).await {
        Ok((v, tokens)) => {
            // Tokens of this Add and since start (searches included).
            eprintln!("add user={} records={} embed_tokens={tokens} total_embed_tokens={}", req.user_id, v.len(), embed::TOKENS_USED.load(std::sync::atomic::Ordering::Relaxed));
            v
        }
        Err(e) => {
            eprintln!("add user={} embedding failed: {e}", req.user_id);
            return error(StatusCode::SERVICE_UNAVAILABLE, e);
        }
    };
    let records = rows
        .iter()
        .zip(vectors)
        .map(|((id, raw, _, content, created_at), embedding)| Record {
            user_tokens: user_parts(content).0,
            instructions: user_parts(content).1,
            id: id.clone(),
            speaker: speaker_of(raw),
            tokens: search::tokens(content),
            content: content.clone(),
            created_at: created_at.clone(),
            embedding,
        })
        .collect::<Vec<_>>();
    {
        let mut db = state.db.lock().unwrap();
        let tx = match db.transaction() {
            Ok(tx) => tx,
            Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        };
        let result = (|| -> rusqlite::Result<()> {
            let next_seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq), 0) + 1 FROM records", [], |r| r.get(0))?;
            for (i, (r, (_, raw, _, _, _))) in records.iter().zip(rows.iter()).enumerate() {
                let blob = r.embedding.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                tx.execute(
                    "INSERT INTO records (id, user_id, session_id, request_id, seq, raw, content, created_at, embedding) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                    params![r.id, req.user_id, req.session_id, req.request_id, next_seq + i as i64, raw, r.content, r.created_at, blob],
                )?;
            }
            tx.execute("INSERT INTO user_requests (user_id, request_id, session_id) VALUES (?1, ?2, ?3)", params![req.user_id, req.request_id, req.session_id])?;
            Ok(())
        })();
        if let Err(e) = result.and_then(|_| tx.commit()) {
            return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
        }
    }
    // Loaded after the commit, a user not in memory comes back with these
    // records already; one in memory gets them appended.
    let cached = state.cache.lock().unwrap().get(&req.user_id).map(|c| c.user.clone());
    match cached {
        Some(user) => {
            let mut user = user.lock().unwrap();
            for record in records {
                user.push(&req.user_id, record);
            }
        }
        None => {
            if let Err(e) = user_state(&state, &req.user_id).await {
                return error(StatusCode::INTERNAL_SERVER_ERROR, e);
            }
        }
    }
    evict(&mut state.cache.lock().unwrap(), state.cache_budget);
    ok()
}

/// Behaviour changes that await a paired answer check before they enter the
/// evaluated configuration; each defaults to the smoke-tested behaviour.
/// TIDEMARK_HEADING_WORDS=on: "Reminder: …", "Note: …" and similar headings are
/// not taken for a speaker's name, so the message keeps its role.
fn heading_words() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("TIDEMARK_HEADING_WORDS").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")))
}

/// Words that open a message as a heading, never as a name.
const HEADING_WORDS: &[&str] = &["reminder", "note", "notes", "update", "important", "warning", "tip", "example", "summary", "edit", "fyi", "todo", "disclaimer", "caution"];

/// The speaker a message names at its start ("Caroline: …"), lower case.
fn speaker_of(text: &str) -> Option<String> {
    speaker_of_with(text, heading_words())
}

fn speaker_of_with(text: &str, skip_headings: bool) -> Option<String> {
    let (head, _) = text.split_once(':')?;
    let head = head.trim();
    let name = head.to_lowercase();
    (!head.is_empty() && head.len() <= 40 && head.split_whitespace().count() <= 4 && !head.contains(['.', '?', '!'])
        && !(skip_headings && HEADING_WORDS.contains(&name.as_str())))
        .then_some(name)
}

fn short_hash(text: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(&Sha256::digest(text.as_bytes())[..8])
}

async fn search_handler(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Result<Json<SearchRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let _permit = match permit(&state) {
        Ok(p) => p,
        Err(busy) => return busy,
    };
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let mut query = content_text(&req.query);
    for option in &req.options {
        query.push(' ');
        query.push_str(&content_text(option));
    }
    if req.top_k == 0 {
        return Json(serde_json::json!({"data": []})).into_response();
    }
    let qv = match state.embedder.embed(&[query.clone()]).await {
        Ok(mut v) => v.remove(0),
        Err(e) => {
            eprintln!("search user={} embedding failed: {e}", req.user_id);
            return error(StatusCode::SERVICE_UNAVAILABLE, e);
        }
    };
    // The user's lock keeps Adds out while ranking; their state is loaded
    // from SQLite if it is not in memory.
    let lock = user_lock(&state, &req.user_id);
    let _guard = lock.lock().await;
    let user = match user_state(&state, &req.user_id).await {
        Ok(u) => u,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e),
    };
    let settings = state.settings.clone();
    let reranker = state.reranker.clone();
    let limit = match req.max_results.unwrap_or(settings.max_results) {
        0 => req.top_k,
        n => n.min(req.top_k),
    };
    let max_chars = req.max_chars.unwrap_or(settings.max_chars);
    let user_weight = req.user_weight.unwrap_or(settings.user_weight).max(0.0);
    let instruction_slots = req.instruction_slots.unwrap_or(settings.instruction_slots);
    let rerank_candidates = req.rerank_candidates.unwrap_or(settings.rerank_candidates);
    let rerank_all = req.rerank_scope.as_deref().map_or(settings.rerank_all, |v| v.trim().eq_ignore_ascii_case("all"));
    // Ranking is CPU work on the user's records, done in place without copying them.
    let data = tokio::task::block_in_place(|| {
        let mut guard = user.lock().unwrap();
        let user = &mut *guard;
        // With the memory engine, its ranking comes first; the hybrid ranking
        // fills the remaining slots (the engine keeps only what it judges
        // relevant, usually far fewer than top_k).
        let engine_first: Vec<String> = match &mut user.engine {
            Some(engine) => {
                let hits = engine.engine.search(&req.user_id, &query, req.top_k.max(rerank_candidates).saturating_mul(3));
                let mut seen = std::collections::HashSet::new();
                hits.into_iter()
                    .filter_map(|hit| engine.records.get(&hit.id).map(|(id, _)| id.clone()))
                    .filter(|id| seen.insert(id.clone()))
                    .take(req.top_k.max(rerank_candidates))
                    .collect()
            }
            None => Vec::new(),
        };
        let records = &user.records;
        if records.is_empty() {
            return Vec::new();
        }
        let bm = search::Bm25::new(records.iter().map(|r| r.tokens.as_slice()).collect());
        let lexical = bm.scores(&search::tokens(&query));
        let vector = records.iter().map(|r| embed::cosine(&qv, &r.embedding)).collect::<Vec<_>>();
        let mut fused = search::fuse(&lexical, &vector, settings.lexical_weight, settings.rrf_k);
        // The user's own lines ranked on their own: questions are mostly about
        // what the user said, which long replies around it would outweigh.
        let query_tokens = search::tokens(&query);
        if user_weight > 0.0 && records.iter().any(|r| !r.user_tokens.is_empty()) {
            let ubm = search::Bm25::new(records.iter().map(|r| r.user_tokens.as_slice()).collect());
            let ulex = ubm.scores(&query_tokens);
            // Records without user lines get no vector rank.
            let uvec = records.iter().zip(&vector).map(|(r, v)| if r.user_tokens.is_empty() { -2.0 } else { *v }).collect::<Vec<_>>();
            let mut extra = vec![0.0f32; records.len()];
            for (i, s) in search::fuse(&ulex, &uvec, settings.lexical_weight, settings.rrf_k) {
                extra[i] = s;
            }
            for (i, score) in fused.iter_mut() {
                *score += user_weight * extra[*i];
            }
            fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
        // Standing instructions and preferences, ranked among themselves.
        let instructions: Vec<(usize, &str)> = if instruction_slots > 0 {
            let all = records.iter().enumerate().flat_map(|(i, r)| r.instructions.iter().map(move |t| (i, t.as_str()))).collect::<Vec<_>>();
            let toks = all.iter().map(|(_, t)| search::tokens(t)).collect::<Vec<_>>();
            let ibm = search::Bm25::new(toks.iter().map(|t| t.as_slice()).collect());
            let ilex = ibm.scores(&query_tokens);
            let ivec = all.iter().map(|(i, _)| vector[*i]).collect::<Vec<_>>();
            search::fuse(&ilex, &ivec, settings.lexical_weight, settings.rrf_k).into_iter().map(|(k, _)| all[k]).collect()
        } else {
            Vec::new()
        };
        if settings.speaker_boost > 0.0 {
            let named = search::named_speakers(&query, records.iter().filter_map(|r| r.speaker.as_deref()));
            if !named.is_empty() {
                for (i, score) in fused.iter_mut() {
                    if records[*i].speaker.as_ref().is_some_and(|sp| named.contains(sp)) {
                        *score *= 1.0 + settings.speaker_boost;
                    }
                }
                fused.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        // Cross-encoder reranking of the leading `rerank_candidates`; the rest
        // keep their order behind them, with scores staying descending.
        let rerank = |fused: Vec<(usize, f32)>| -> Vec<(usize, f32)> {
            let (Some(reranker), n) = (reranker.as_ref(), rerank_candidates.min(fused.len())) else { return fused };
            if n == 0 {
                return fused;
            }
            let passages = fused[..n].iter().map(|(i, _)| records[*i].content.as_str()).collect::<Vec<_>>();
            let (pairs0, cut0) = (rerank::PAIRS.load(std::sync::atomic::Ordering::Relaxed), rerank::TRUNCATED.load(std::sync::atomic::Ordering::Relaxed));
            let result = reranker.scores(&query, &passages);
            eprintln!(
                "rerank user={} pairs={} truncated={} (since start: pairs={} truncated={})",
                req.user_id,
                rerank::PAIRS.load(std::sync::atomic::Ordering::Relaxed) - pairs0,
                rerank::TRUNCATED.load(std::sync::atomic::Ordering::Relaxed) - cut0,
                rerank::PAIRS.load(std::sync::atomic::Ordering::Relaxed),
                rerank::TRUNCATED.load(std::sync::atomic::Ordering::Relaxed)
            );
            match result {
                Ok(scores) => {
                    let mut head = fused[..n].iter().zip(scores).map(|((i, _), s)| (*i, s)).collect::<Vec<_>>();
                    head.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    let floor = head.last().map(|r| r.1).unwrap_or(0.0);
                    let tail = fused[n..].iter().enumerate().map(|(k, (i, _))| (*i, floor - 1.0 - k as f32 * 1e-3));
                    head.into_iter().chain(tail).collect()
                }
                Err(e) => {
                    eprintln!("rerank failed, order kept: {e}");
                    fused
                }
            }
        };
        if !rerank_all {
            fused = rerank(fused);
        }
        if !engine_first.is_empty() {
            let position = records.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect::<HashMap<_, _>>();
            let head = engine_first.iter().filter_map(|id| position.get(id.as_str()).copied()).collect::<Vec<_>>();
            let head_set = head.iter().copied().collect::<std::collections::HashSet<_>>();
            let tail = fused.iter().map(|(i, _)| *i).filter(|i| !head_set.contains(i)).collect::<Vec<_>>();
            // Strictly descending scores in the combined order.
            let n = (head.len() + tail.len()) as f32;
            fused = head.into_iter().chain(tail).enumerate().map(|(rank, i)| (i, (n - rank as f32) / n)).collect();
        }
        if rerank_all {
            fused = rerank(fused);
        }
        // Instructions take the last slots, after the ranked records, so the
        // ranking itself is unchanged; the content budget is shared. At most a
        // tenth of the results are instructions (10 of AML's 100). With
        // TIDEMARK_INSTRUCTION_DEDUP=on, an instruction whose record is already
        // among the results is skipped and its slot goes to the next record.
        let slots = instruction_slots.min(limit / 10).min(instructions.len());
        let dedupe = std::env::var("TIDEMARK_INSTRUCTION_DEDUP").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on"));
        let chosen: Vec<(usize, &str)> = if dedupe {
            let main = fused.iter().take(limit - slots).map(|(i, _)| *i).collect::<std::collections::HashSet<_>>();
            instructions.iter().filter(|(i, _)| !main.contains(i)).take(slots).copied().collect()
        } else {
            instructions.iter().take(slots).copied().collect()
        };
        let instruction_items = chosen.iter().enumerate().map(|(j, (i, text))| {
            let r = &records[*i];
            let header = r.content.strip_prefix('[').and_then(|c| c.split_once(']')).map(|(h, _)| format!(", {h}")).unwrap_or_default();
            (format!("{}#i{j}", r.id), format!("[standing instruction{header}] user: {text}"), *i)
        }).collect::<Vec<_>>();
        let reserved_chars = instruction_items.iter().map(|(_, c, _)| c.chars().count()).sum::<usize>();
        let mut chars = 0usize;
        let mut items = fused
            .into_iter()
            .take(limit - instruction_items.len())
            .enumerate()
            .take_while(|(rank, (i, _))| {
                chars += records[*i].content.chars().count();
                max_chars == 0 || *rank == 0 || chars + reserved_chars <= max_chars
            })
            .map(|(_, (i, score))| (records[i].id.clone(), records[i].content.clone(), score, i))
            .collect::<Vec<_>>();
        let last = items.last().map_or(1.0, |it| it.2);
        items.extend(instruction_items.into_iter().enumerate().map(|(j, (id, content, i))| (id, content, last - (j + 1) as f32 * 1e-3, i)));
        items
            .into_iter()
            .map(|(id, content, score, i)| {
                let mut item = serde_json::json!({"id": id, "content": content, "score": score});
                if let Some(c) = &records[i].created_at {
                    item["created_at"] = serde_json::json!(c);
                }
                item
            })
            .collect::<Vec<_>>()
    });
    Json(serde_json::json!({"data": data})).into_response()
}

fn open_db(path: &str) -> rusqlite::Result<Connection> {
    let db = Connection::open(path)?;
    db.execute_batch(
        "PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
         CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL,
            request_id TEXT NOT NULL, seq INTEGER NOT NULL, raw TEXT NOT NULL, content TEXT NOT NULL, created_at TEXT, embedding BLOB NOT NULL);
         CREATE INDEX IF NOT EXISTS records_user ON records(user_id, session_id, seq);
         CREATE INDEX IF NOT EXISTS records_seq ON records(seq);
         CREATE TABLE IF NOT EXISTS requests (request_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, session_id TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
         CREATE TABLE IF NOT EXISTS user_requests (user_id TEXT NOT NULL, request_id TEXT NOT NULL, session_id TEXT NOT NULL,
            PRIMARY KEY (user_id, request_id));
         INSERT OR IGNORE INTO user_requests (user_id, request_id, session_id) SELECT user_id, request_id, session_id FROM requests;",
    )?;
    Ok(db)
}


/// Stored vectors must come from the configured embedding model and have its
/// dimensions. The model and dimensions are recorded when the first database
/// opens empty. A database with records but no record of their model (made by
/// an older version) is not stamped: its vectors' dimensions must match, a
/// warning is logged, and TIDEMARK_ADOPT_EMBEDDINGS=1 records the current
/// configuration once its provenance is known.
async fn check_embedding_provenance(db: &Connection, embedder: &embed::Embedder, records: i64) -> Result<(), String> {
    let model = embedder.label();
    let dims = match embedder {
        embed::Embedder::Hash { dims } => *dims,
        embed::Embedder::Remote { .. } if std::env::var_os("TIDEMARK_SKIP_EMBED_CHECK").is_some() => {
            eprintln!("tidemark: WARNING: embedding check skipped, so stored vectors are not checked against the model");
            return Ok(());
        }
        embed::Embedder::Remote { .. } => {
            embedder.embed(&["dimension check".to_string()]).await?.first().map(Vec::len).ok_or("embedding returned no vector")?
        }
    };
    let stored = {
        let mut stmt = db.prepare("SELECT DISTINCT length(embedding) / 4 FROM records").map_err(|e| e.to_string())?;
        let rows = stmt.query_map([], |r| r.get::<_, i64>(0)).map_err(|e| e.to_string())?;
        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())?
    };
    if stored.len() > 1 {
        return Err(format!("stored vectors have mixed dimensions {stored:?}"));
    }
    if let Some(&d) = stored.first() {
        if d as usize != dims {
            return Err(format!("stored vectors have {d} dimensions, the configured embedding ({model}) has {dims}"));
        }
    }
    let get = |k: &str| db.query_row("SELECT value FROM meta WHERE key = ?1", [k], |r| r.get::<_, String>(0)).ok();
    let set = |k: &str, v: &str| db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![k, v]).map(|_| ()).map_err(|e| e.to_string());
    match (get("embedding_model"), get("embedding_dims")) {
        (Some(m), Some(d)) => {
            if m != model || d != dims.to_string() {
                return Err(format!("database was built with {m} ({d} dimensions), the configured embedding is {model} ({dims} dimensions)"));
            }
            eprintln!("tidemark: embeddings {model} ({dims} dimensions) match the database");
        }
        _ if records == 0 || std::env::var_os("TIDEMARK_ADOPT_EMBEDDINGS").is_some() => {
            set("embedding_model", &model)?;
            set("embedding_dims", &dims.to_string())?;
            eprintln!("tidemark: recorded embeddings {model} ({dims} dimensions) for this database");
        }
        _ => eprintln!(
            "tidemark: WARNING: the database's {records} records do not record their embedding model; dimensions match ({dims}), \
             but the model cannot be verified. Not recorded; set TIDEMARK_ADOPT_EMBEDDINGS=1 once the provenance is known."
        ),
    }
    Ok(())
}

/// Ctrl-C or SIGTERM (as sent by systemd) starts a graceful shutdown.
async fn shutdown_signal() {
    let interrupt = async { let _ = tokio::signal::ctrl_c().await; };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => { s.recv().await; }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! { _ = interrupt => {}, _ = terminate => {} }
}

/// One line per request: time, method, path, status, duration (no content, no keys).
async fn access_log(req: axum::extract::Request, next: axum::middleware::Next) -> axum::response::Response {
    let started = std::time::Instant::now();
    let (method, path) = (req.method().clone(), req.uri().path().to_string());
    let response = next.run(req).await;
    if path != "/health" || !response.status().is_success() {
        eprintln!("{} {method} {path} {} {}ms", Utc::now().format("%Y-%m-%dT%H:%M:%SZ"), response.status().as_u16(), started.elapsed().as_millis());
    }
    response
}

#[tokio::main]
async fn main() {
    let db_path = std::env::var("TIDEMARK_DB").unwrap_or_else(|_| "tidemark.sqlite".into());
    let port = std::env::var("TIDEMARK_PORT").unwrap_or_else(|_| "18200".into());
    let host = std::env::var("TIDEMARK_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let db = open_db(&db_path).expect("open database");
    let (users, records): (i64, i64) = db
        .query_row("SELECT COUNT(DISTINCT user_id), COUNT(*) FROM records", [], |r| Ok((r.get(0)?, r.get(1)?)))
        .expect("count records");
    let keys = std::env::var("TIDEMARK_API_KEYS").unwrap_or_default().split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect::<Vec<_>>();
    let embedder = embed::Embedder::from_env();
    // A wrong key or URL would fail every Add and Search; refuse to start instead
    // (TIDEMARK_SKIP_EMBED_CHECK=1 skips this, e.g. for offline tests).
    if matches!(embedder, embed::Embedder::Remote { .. }) && std::env::var_os("TIDEMARK_SKIP_EMBED_CHECK").is_none() {
        match embedder.embed(&["startup check".to_string()]).await {
            Ok(v) => eprintln!("tidemark: embedding check ok ({} dimensions)", v.first().map(Vec::len).unwrap_or(0)),
            Err(e) => {
                eprintln!("tidemark: embedding check failed, not starting: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Err(e) = check_embedding_provenance(&db, &embedder, records).await {
        eprintln!("tidemark: {e}; not starting");
        std::process::exit(1);
    }
    let reranker = rerank::Reranker::from_env().expect("load reranker").map(Arc::new);
    let use_engine = matches!(std::env::var("TIDEMARK_BACKEND").unwrap_or_else(|_| "engine".into()).trim(), "engine" | "");
    let cache_mb = std::env::var("TIDEMARK_CACHE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(2048).max(16);
    let st = Settings::from_env();
    eprintln!(
        "tidemark {}: window {}, max words {}, lexical weight {}, rrf k {}, user weight {}, instruction slots {}, max results {}, max chars {}, max concurrent {}",
        env!("CARGO_PKG_VERSION"),
        st.window,
        st.max_words,
        st.lexical_weight,
        st.rrf_k,
        st.user_weight,
        st.instruction_slots,
        st.max_results,
        st.max_chars,
        std::env::var("TIDEMARK_MAX_CONCURRENT").unwrap_or_else(|_| "8".into())
    );
    eprintln!(
        "tidemark: {users} users, {records} records in {db_path} (loaded per user on demand, cache {cache_mb} MB); backend {}; embeddings {}; rerank {}; auth {}",
        if use_engine { "engine" } else { "builtin" },
        embedder.label(),
        reranker.as_ref().map(|r| r.label.clone()).unwrap_or_else(|| "off".into()),
        if keys.is_empty() { "OFF (set TIDEMARK_API_KEYS)" } else { "on" }
    );
    let state = Arc::new(AppState {
        db: Mutex::new(db),
        cache: Mutex::new(HashMap::new()),
        cache_budget: cache_mb * 1024 * 1024,
        use_engine,
        clock: std::sync::atomic::AtomicU64::new(0),
        add_locks: Mutex::new(HashMap::new()),
        permits: Arc::new(tokio::sync::Semaphore::new(
            std::env::var("TIDEMARK_MAX_CONCURRENT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(8usize).max(1),
        )),
        embedder,
        reranker,
        keys,
        settings: Settings::from_env(),
    });
    let app = Router::new()
        .route("/health", get(|| async { Json(serde_json::json!({"status": "ok", "service": "tidemark", "version": env!("CARGO_PKG_VERSION")})) }))
        .route("/add", post(add))
        .route("/search", post(search_handler))
        // Add requests may carry long messages or base64 images (up to 30 MB).
        .layer(axum::extract::DefaultBodyLimit::max(64 * 1024 * 1024))
        // Gzip responses when the client accepts it: a 100-record Search answer
        // is mostly text and shrinks several times over slow links.
        .layer(tower_http::compression::CompressionLayer::new().gzip(true))
        .layer(axum::middleware::from_fn(access_log))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(format!("{host}:{port}")).await.expect("bind");
    eprintln!("tidemark listening on http://{host}:{port}");
    axum::serve(listener, app).with_graceful_shutdown(shutdown_signal()).await.expect("serve");
    eprintln!("tidemark stopped");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_carries_its_date_speaker_time_notes_and_reply_context() {
        let s = Settings { max_words: 300, previous_context_chars: 20, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 1, rerank_candidates: 0, rerank_all: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false };
        // 8 May 2023 13:56 UTC
        let ts = Some(1683554160000);
        let out = render("user", "Caroline: I went to a support group yesterday.", ts, Some("Melanie: How have you been lately, Caroline?"), &s);
        assert_eq!(out[0].0, "[Monday 8 May 2023, 13:56] Caroline: I went to a support group yesterday. [time: yesterday = 7 May 2023] (in reply to: Melanie: How have yo…)");
        let plain = render("assistant", "Sure, here it is.", None, None, &s);
        assert_eq!(plain[0].0, "assistant: Sure, here it is.");
    }

    #[test]
    fn messages_are_grouped_into_non_overlapping_dated_blocks() {
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 2, rerank_candidates: 0, rerank_all: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false };
        let ts = Some(1683554160000);
        let m = |t: &str| ("user".to_string(), t.to_string(), ts);
        let long = format!("Caroline: {}", "word ".repeat(400));
        let blocks = render_blocks(&[m("Caroline: I went yesterday."), m("Melanie: Nice!"), m("Caroline: Thanks."), m(&long), m("Melanie: Wow.")], &s);
        assert_eq!(blocks[0].1, "[Monday 8 May 2023, 13:56] \nCaroline: I went yesterday. [time: yesterday = 7 May 2023]\nMelanie: Nice!");
        assert_eq!(blocks[1].1, "[Monday 8 May 2023, 13:56] \nCaroline: Thanks.", "a long message closes the block");
        assert_eq!(blocks.len(), 5, "the long message is split on its own (2 parts), then the last block");
        assert_eq!(blocks[4].1, "[Monday 8 May 2023, 13:56] \nMelanie: Wow.");
    }

    #[test]
    fn headings_are_not_names_when_enabled_and_names_stay_names() {
        for name in ["Caroline: I went yesterday", "Melanie: hi", "Dr Smith: take two", "Mom: call me", "John Paul: ok"] {
            assert!(speaker_of_with(name, true).is_some(), "{name}");
            assert_eq!(speaker_of_with(name, true), speaker_of_with(name, false));
        }
        for heading in ["Reminder: always use metric units", "Note: the meeting moved", "Important: deadline Friday", "TODO: fix tests"] {
            assert!(speaker_of_with(heading, true).is_none(), "{heading}");
            assert!(speaker_of_with(heading, false).is_some(), "default keeps the evaluated behaviour: {heading}");
        }
    }

    #[test]
    fn blocks_split_on_a_new_day_only() {
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false };
        let base = 1704168000000i64; // 2024-01-02 04:00 UTC
        let m = |t: &str, ts: i64| ("user".to_string(), t.to_string(), Some(ts));
        // AML-style: one session, messages five minutes apart -> one block, as before
        let same_day = (0..5).map(|i| m(&format!("message {i}"), base + i * 300_000)).collect::<Vec<_>>();
        let blocks = render_blocks(&same_day, &s);
        assert_eq!(blocks.len(), 1);
        assert!(blocks[0].1.starts_with("[Tuesday 2 January 2024, 04:00]"));
        // one timestamp for all -> one block
        assert_eq!(render_blocks(&(0..5).map(|i| m(&format!("m{i}"), base)).collect::<Vec<_>>(), &s).len(), 1);
        // a message from the next day starts a new block with its own date
        let mut two_days = same_day[..3].to_vec();
        two_days.push(m("next day", base + 86_400_000));
        let blocks = render_blocks(&two_days, &s);
        assert_eq!(blocks.len(), 2);
        assert!(blocks[1].1.starts_with("[Wednesday 3 January 2024, 04:00]"), "{}", blocks[1].1);
    }

    #[test]
    fn every_part_of_a_split_message_names_its_speaker() {
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false };
        let long = format!("Caroline: {}", (0..700).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" "));
        let off = render("user", &long, Some(1683554160000), None, &s);
        assert!(off.len() >= 3 && off[0].0.contains("] Caroline: ") && !off[1].0.contains("Caroline: "), "default: as evaluated");
        let s = Settings { part_speakers: true, ..s };
        let parts = render("user", &long, Some(1683554160000), None, &s);
        assert!(parts.len() >= 3);
        assert!(parts.iter().all(|(c, _)| c.contains("] Caroline: ")), "{:?}", parts.iter().map(|(c, _)| c.chars().take(50).collect::<String>()).collect::<Vec<_>>());
        let plain = render("user", &"word ".repeat(700), None, None, &s);
        assert!(plain.iter().all(|(c, _)| c.starts_with("user: ")));
    }

    #[test]
    fn split_messages_embed_each_part_once_with_its_speaker() {
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false };
        assert_eq!(embed_texts("user", "Caroline: short one", &s), ["Caroline: short one"], "a message in one record is embedded as is");
        let long = format!("Caroline: {}", (0..700).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" "));
        let parts = embed_texts("user", &long, &s);
        assert!(parts.len() >= 3);
        assert!(parts.iter().all(|p| p.starts_with("Caroline: ") && p.len() < long.len() / 2), "{:?}", parts.iter().map(|p| p.len()).collect::<Vec<_>>());
        assert!(parts.last().unwrap().ends_with("w699"));
        let plain = embed_texts("assistant", &"word ".repeat(700), &s);
        assert!(plain.iter().all(|p| p.starts_with("assistant: ")));
        let blocks = render_blocks(&[("user".into(), long.clone(), None)], &s);
        assert_eq!(blocks.len(), parts.len());
        assert!(blocks.iter().zip(&parts).all(|(b, p)| &b.0 == p), "split records store and embed their own part");
    }

    #[test]
    fn text_without_spaces_is_split_by_characters() {
        // e.g. Chinese: one "word" by whitespace, far beyond an embedding input limit
        let text = "记忆".repeat(5000);
        let parts = split_words(&text, 300);
        assert!(parts.len() >= 5, "{}", parts.len());
        assert!(parts.iter().all(|p| p.chars().count() <= MAX_PART_CHARS), "every part fits");
        let joined_chars: usize = parts.iter().map(|p| p.chars().filter(|c| *c != ' ').count()).sum();
        assert!(joined_chars >= 10000, "nothing is lost (parts overlap)");
    }

    #[test]
    fn short_text_is_kept_whole_and_long_text_always_progresses() {
        assert_eq!(split_words("A short message.", 300), ["A short message."]);
        let one_long_word = "x".repeat(1999);
        assert_eq!(split_words(&one_long_word, 300).len(), 1);
        let many = (0..5000).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let parts = split_words(&many, 300);
        assert!(parts.last().unwrap().ends_with("w4999"));
    }

    #[test]
    fn long_messages_are_split_with_overlap() {
        let text = (0..700).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
        let parts = split_words(&text, 300);
        assert_eq!(parts.len(), 3);
        assert!(parts[1].starts_with("w240 "));
        assert!(parts[2].ends_with("w699"));
    }
}
