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

mod api;
mod applies;
mod assembly;
mod followups;
mod dates;
mod passages;
mod embed;
mod http;
mod llm;
mod preferences;
mod reembed;
mod rerank;
mod rules;
mod search;
mod units;

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
    /// The user's sentences that state a change, stop, resumption or exception (for evidence assembly).
    changes: Vec<String>,
}

/// The user-line tokens and standing instructions of a record's content.
fn user_parts(content: &str) -> (Vec<String>, Vec<String>, Vec<String>) {
    let lines = search::user_lines(content);
    let tokens = lines.iter().flat_map(|l| search::tokens(l)).collect();
    let instructions = lines.into_iter().filter(|l| search::is_standing_instruction(l)).map(|l| l.chars().take(1000).collect()).collect();
    (tokens, instructions, assembly::change_sentences(content))
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
    /// The optional chat model (`TIDEMARK_LLM_*`); Add and Search never call
    /// it (preference extraction by the model runs after Add, in the background).
    llm: Option<Arc<tidemark_engine::ChatModel>>,
    preferences: preferences::Config,
    /// Users whose queued preference extraction is being processed.
    preference_busy: Mutex<std::collections::HashSet<String>>,
    preference_wake: tokio::sync::Notify,
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
    /// BM25 statistics over the records' text and over their user lines,
    /// kept up to date as records are added.
    bm25: search::Bm25Index,
    user_bm25: search::Bm25Index,
    /// Smaller retrieval units (TIDEMARK_UNITS=on), when built for this user.
    units: Option<units::UnitIndex>,
}

impl UserState {
    fn push(&mut self, user_id: &str, record: Record) {
        if let Some(engine) = &mut self.engine {
            engine.store(user_id, &record);
        }
        self.bytes.fetch_add(record_bytes(&record), std::sync::atomic::Ordering::Relaxed);
        self.bm25.add(&record.tokens);
        self.user_bm25.add(&record.user_tokens);
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
    /// Evidence assembly (TIDEMARK_ASSEMBLY=on): at most this many related change records (0 = off).
    assembly_max: usize,
    /// Share of the user's records a term may occur in and still count as rare for assembly.
    assembly_rare: f32,
    /// Characteristic terms per returned record that define its topic for assembly.
    assembly_terms: usize,
    /// Smaller retrieval units (TIDEMARK_UNITS=on): loaded with users and used by default.
    units: bool,
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
    /// Slots at the end of the results for the user's lines that the cue rules flag as standing instructions or
    /// preferences, most relevant first (0 = off, the default since 1.2; 1.0.0 used 10). Opt-in because the cue
    /// rules also flag ordinary text (pasted examples, narrative "I prefer …"), which would be shown as the user's
    /// standing instruction.
    instruction_slots: usize,
    /// Every part of a split named message repeats the speaker
    /// (TIDEMARK_PART_SPEAKERS=on; off = as evaluated).
    part_speakers: bool,
    /// Follow-ups (TIDEMARK_FOLLOWUPS): at most this many of the user's statements from other times attached to
    /// the returned records they relate to (0 = off), and how they are chosen.
    followups: followups::Options,
    /// When a statement applies (TIDEMARK_APPLIES=on): the period a returned user sentence states for itself,
    /// resolved against its record's date, as a note and an `applies` field. Computed at Search; nothing stored.
    applies: bool,
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
            assembly_max: if std::env::var("TIDEMARK_ASSEMBLY").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")) { num("TIDEMARK_ASSEMBLY_MAX", 3.0) as usize } else { 0 },
            assembly_rare: num("TIDEMARK_ASSEMBLY_RARE", 0.15) as f32,
            assembly_terms: num("TIDEMARK_ASSEMBLY_TERMS", 8.0) as usize,
            units: std::env::var("TIDEMARK_UNITS").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")),
            rerank_all: std::env::var("TIDEMARK_RERANK_SCOPE").is_ok_and(|v| v.trim().eq_ignore_ascii_case("all")),
            max_results: num("TIDEMARK_MAX_RESULTS", 0.0) as usize,
            max_chars: num("TIDEMARK_MAX_CHARS", 0.0) as usize,
            user_weight: num("TIDEMARK_USER_WEIGHT", 1.0).max(0.0),
            instruction_slots: num("TIDEMARK_INSTRUCTION_SLOTS", 0.0) as usize,
            part_speakers: std::env::var("TIDEMARK_PART_SPEAKERS").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")),
            followups: followups::Options {
                max: num("TIDEMARK_FOLLOWUPS", 0.0) as usize,
                rare: num("TIDEMARK_FOLLOWUP_RARE", 0.05),
                min_terms: num("TIDEMARK_FOLLOWUP_MIN_TERMS", 0.0) as usize,
                any_time: std::env::var("TIDEMARK_FOLLOWUP_SCOPE").is_ok_and(|v| v.trim().eq_ignore_ascii_case("any")),
                anchor_share: num("TIDEMARK_FOLLOWUP_ANCHOR_SHARE", 0.5).clamp(0.0, 1.0),
                max_chars: num("TIDEMARK_FOLLOWUP_CHARS", 400.0) as usize,
            },
            applies: std::env::var("TIDEMARK_APPLIES").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")),
        }
    }
}

#[derive(Deserialize)]
struct Message {
    #[serde(default)]
    role: String,
    #[serde(default)]
    timestamp: Option<i64>,
    /// The sender's offset from UTC in minutes at the time of the message (120 = UTC+02:00), optional. With it,
    /// the record's date header and its time notes ("yesterday", "tomorrow") use the sender's calendar day; the
    /// stored `created_at` stays the UTC instant. Without it, everything is in UTC as before. A fixed offset is not
    /// a time zone: it says nothing about daylight-saving changes on other days.
    #[serde(default)]
    utc_offset_minutes: Option<i32>,
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
    /// Per-request override of TIDEMARK_ENGINE_MERGE ("first" or "fused").
    #[serde(default)]
    engine_merge: Option<String>,
    /// Per-request override of TIDEMARK_SEARCH_RULES (the user's active explicit rules in a separate `rules` field),
    /// and a scope that limits them to that scope plus unscoped rules.
    #[serde(default)]
    rules: Option<bool>,
    #[serde(default)]
    rule_scope: Option<String>,
    /// Per-request overrides of TIDEMARK_PASSAGES, TIDEMARK_PASSAGE_MAX_CHARS and TIDEMARK_PASSAGE_MIN_WORDS.
    #[serde(default)]
    passages: Option<bool>,
    #[serde(default)]
    passage_max_chars: Option<usize>,
    #[serde(default)]
    passage_min_words: Option<usize>,
    /// Per-request overrides of TIDEMARK_PREFERENCE_SLOTS and TIDEMARK_PREFERENCE_LABELS.
    #[serde(default)]
    preference_slots: Option<usize>,
    #[serde(default)]
    preference_labels: Option<String>,
    /// Per-request switch and cap for evidence assembly (TIDEMARK_ASSEMBLY, TIDEMARK_ASSEMBLY_MAX).
    #[serde(default)]
    assembly: Option<bool>,
    #[serde(default)]
    assembly_max: Option<usize>,
    /// Per-request overrides of TIDEMARK_FOLLOWUPS, TIDEMARK_FOLLOWUP_SCOPE ("later" or "any"),
    /// TIDEMARK_FOLLOWUP_RARE, TIDEMARK_FOLLOWUP_MIN_TERMS, TIDEMARK_FOLLOWUP_CHARS and
    /// TIDEMARK_FOLLOWUP_ANCHOR_SHARE.
    #[serde(default)]
    followups: Option<usize>,
    #[serde(default)]
    followup_scope: Option<String>,
    #[serde(default)]
    followup_rare: Option<f32>,
    #[serde(default)]
    followup_min_terms: Option<usize>,
    #[serde(default)]
    followup_chars: Option<usize>,
    #[serde(default)]
    followup_anchor_share: Option<f32>,
    /// Per-request override of TIDEMARK_APPLIES.
    #[serde(default)]
    applies: Option<bool>,
    /// Per-request switch for smaller retrieval units (effective when TIDEMARK_UNITS=on loaded them).
    #[serde(default)]
    units: Option<bool>,
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

/// The header, calendar day and UTC instant of a message sent at a known offset from UTC: the header and the day
/// are the sender's local ones, marked with the offset; the instant (`created_at`) is unchanged. Without an offset
/// this is `date_header`.
fn date_header_at(ts: Option<i64>, offset: Option<i32>) -> (String, Option<chrono::NaiveDate>, Option<String>) {
    let Some(offset) = offset else { return date_header(ts) };
    match ts.and_then(DateTime::<Utc>::from_timestamp_millis) {
        Some(dt) => {
            let local = dt.naive_utc() + chrono::Duration::minutes(offset as i64);
            let sign = if offset < 0 { '-' } else { '+' };
            (
                format!("[{} UTC{sign}{:02}:{:02}] ", local.format("%A %-d %B %Y, %H:%M"), offset.abs() / 60, offset.abs() % 60),
                Some(local.date()),
                Some(dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
            )
        }
        None => (String::new(), None, None),
    }
}

/// The largest offsets in use are UTC−12:00 and UTC+14:00.
fn valid_offset(offset: i32) -> bool {
    (-12 * 60..=14 * 60).contains(&offset)
}

#[cfg(test)]
fn render(role: &str, text: &str, ts: Option<i64>, previous: Option<&str>, s: &Settings) -> Vec<(String, Option<String>)> {
    render_at(role, text, ts, None, previous, s)
}

/// The records of one message, as the reader will see them.
fn render_at(role: &str, text: &str, ts: Option<i64>, offset: Option<i32>, previous: Option<&str>, s: &Settings) -> Vec<(String, Option<String>)> {
    let (header, date, created_at) = date_header_at(ts, offset);
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
        let (user_tokens, instructions, changes) = user_parts(&content);
        Ok(Record {
            user_tokens,
            instructions,
            changes,
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
        bm25: search::Bm25Index::default(),
        user_bm25: search::Bm25Index::default(),
        units: None,
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
        let mut user = build_user(&id, records, st.use_engine);
        if st.settings.units {
            let db = st.db.lock().unwrap();
            user.units = units::load(&db, &id, &user.records).map_err(|e| e.to_string())?;
        }
        Ok(user)
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
fn block_line(role: &str, text: &str, ts: Option<i64>, offset: Option<i32>) -> String {
    let (_, date, _) = date_header_at(ts, offset);
    let speaker = if speaker_of(text).is_some() || role.trim().is_empty() { String::new() } else { format!("{}: ", role.trim()) };
    let notes = date.map(|d| dates::annotate(text, d)).unwrap_or_default();
    let notes = if notes.is_empty() { String::new() } else { format!(" [time: {}]", notes.join("; ")) };
    format!("{speaker}{}{notes}", text.trim())
}

/// Records for consecutive messages grouped `window` at a time: one date
/// header (the first message's) and one line per message. A message longer
/// than `max_words` is left out of blocks and split on its own.
#[cfg(test)]
fn render_blocks(messages: &[(String, String, Option<i64>)], s: &Settings) -> Vec<(String, String, Option<String>)> {
    render_blocks_at(messages, &[], s)
}

/// `offsets` gives each message's offset from UTC, by position (missing = none).
fn render_blocks_at(messages: &[(String, String, Option<i64>)], offsets: &[Option<i32>], s: &Settings) -> Vec<(String, String, Option<String>)> {
    type Timed<'a> = (&'a (String, String, Option<i64>), Option<i32>);
    let mut out = Vec::new();
    let mut block: Vec<Timed> = Vec::new();
    let flush = |block: &mut Vec<Timed>, out: &mut Vec<(String, String, Option<String>)>| {
        if block.is_empty() {
            return;
        }
        let (header, _, created_at) = date_header_at(block[0].0.2, block[0].1);
        let lines = block.iter().map(|((role, text, ts), offset)| block_line(role, text, *ts, *offset)).collect::<Vec<_>>();
        let raw = block.iter().map(|((_, text, _), _)| text.trim()).collect::<Vec<_>>().join("\n");
        out.push((raw, format!("{header}\n{}", lines.join("\n")).trim_start().to_string(), created_at));
        block.clear();
    };
    for (i, m) in messages.iter().enumerate() {
        let offset = offsets.get(i).copied().flatten();
        if too_long(&m.1, s.max_words) {
            flush(&mut block, &mut out);
            for ((content, created_at), part) in render_at(&m.0, &m.1, m.2, offset, None, s).into_iter().zip(embed_texts(&m.0, &m.1, s)) {
                out.push((part, content, created_at));
            }
            continue;
        }
        // A block shows one date, so a message from another day starts a new
        // block (messages of the same day keep sharing one). With offsets the
        // day is the sender's, and the header names the offset, so a change of
        // offset also starts a new block.
        if block.first().is_some_and(|(first, first_offset)| date_header_at(first.2, *first_offset).1 != date_header_at(m.2, offset).1 || *first_offset != offset) {
            flush(&mut block, &mut out);
        }
        block.push((m, offset));
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
    if let Some(bad) = req.messages.iter().filter_map(|m| m.utc_offset_minutes).find(|o| !valid_offset(*o)) {
        return error(StatusCode::BAD_REQUEST, format!("utc_offset_minutes must be between -720 and 840, got {bad}"));
    }
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
        let kept = req.messages.iter().map(|m| ((m.role.clone(), content_text(&m.content), m.timestamp), m.utc_offset_minutes))
            .filter(|(m, _)| !m.1.trim().is_empty()).collect::<Vec<_>>();
        let offsets = kept.iter().map(|(_, o)| *o).collect::<Vec<_>>();
        let messages = kept.into_iter().map(|(m, _)| m).collect::<Vec<_>>();
        for (bi, (raw, content, created_at)) in render_blocks_at(&messages, &offsets, &state.settings).into_iter().enumerate() {
            rows.push((format!("{}#b{}", request_key, bi), raw.clone(), raw, content, created_at));
        }
    }
    for (mi, m) in req.messages.iter().enumerate().filter(|_| state.settings.window == 1) {
        let text = content_text(&m.content);
        if text.trim().is_empty() {
            continue;
        }
        let parts = embed_texts(&m.role, &text, &state.settings);
        for (pi, ((content, created_at), part)) in render_at(&m.role, &text, m.timestamp, m.utc_offset_minutes, previous.as_deref(), &state.settings).into_iter().zip(parts).enumerate() {
            let id = format!("{}#{}.{}", request_key, mi, pi);
            // The whole message stays the stored text (the next message's context).
            rows.push((id, text.clone(), part, content, created_at));
        }
        previous = Some(text);
    }
    // Embed the speaker and words; dates are matched lexically.
    let mut embed_inputs = rows.iter().map(|(_, _, embed, _, _)| embed.clone()).collect::<Vec<_>>();
    // The user's own words in each new record, for preference extraction:
    // found by the rules now (embedded with the records), or queued for the model.
    let user_messages = req.messages.iter().filter(|m| m.role.trim().eq_ignore_ascii_case("user")).map(|m| content_text(&m.content)).collect::<Vec<_>>();
    let user_texts = match state.preferences.mode {
        preferences::Mode::Off => Vec::new(),
        _ => preferences::user_texts(&rows.iter().map(|(id, raw, _, _, at)| (id.clone(), raw.clone(), at.clone())).collect::<Vec<_>>(), &user_messages),
    };
    let rule_preferences = match state.preferences.mode {
        preferences::Mode::Rules => user_texts.iter().flat_map(|(id, text, at)| preferences::rule_candidates(id, text, at.as_deref())).collect::<Vec<_>>(),
        _ => Vec::new(),
    };
    embed_inputs.extend(rule_preferences.iter().map(preferences::Candidate::index_text));
    let vectors = match state.embedder.embed_counted(&embed_inputs, embed::Kind::Document).await {
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
    let mut vectors = vectors;
    let preference_vectors = vectors.split_off(rows.len());
    let records = rows
        .iter()
        .zip(vectors)
        .map(|((id, raw, _, content, created_at), embedding)| {
            let (user_tokens, instructions, changes) = user_parts(content);
            Record {
            user_tokens,
            instructions,
            changes,
            id: id.clone(),
            speaker: speaker_of(raw),
            tokens: search::tokens(content),
            content: content.clone(),
            created_at: created_at.clone(),
            embedding,
            }
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
            match state.preferences.mode {
                preferences::Mode::Rules => {
                    preferences::store(&tx, &req.user_id, &rule_preferences.iter().cloned().zip(preference_vectors).collect::<Vec<_>>())?;
                }
                preferences::Mode::Llm => preferences::enqueue(&tx, &req.user_id, &user_texts, state.preferences.delay)?,
                preferences::Mode::Off => {}
            }
            Ok(())
        })();
        if let Err(e) = result.and_then(|_| tx.commit()) {
            return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string());
        }
    }
    if state.preferences.mode == preferences::Mode::Llm && !user_texts.is_empty() {
        state.preference_wake.notify_waiters();
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
    let qv = match state.embedder.embed(&[query.clone()], embed::Kind::Query).await {
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
    let preference_slots = req.preference_slots.unwrap_or(state.preferences.slots);
    let use_units = req.units.unwrap_or(settings.units);
    let assembly_max = match (req.assembly, req.assembly_max) {
        (Some(false), _) => 0,
        (_, Some(n)) => n,
        (Some(true), None) => if settings.assembly_max > 0 { settings.assembly_max } else { 3 },
        (None, None) => settings.assembly_max,
    };
    let followup = followups::Options {
        max: req.followups.unwrap_or(settings.followups.max),
        rare: req.followup_rare.unwrap_or(settings.followups.rare),
        min_terms: req.followup_min_terms.unwrap_or(settings.followups.min_terms),
        any_time: req.followup_scope.as_deref().map_or(settings.followups.any_time, |v| v.trim().eq_ignore_ascii_case("any")),
        max_chars: req.followup_chars.unwrap_or(settings.followups.max_chars),
        anchor_share: req.followup_anchor_share.unwrap_or(settings.followups.anchor_share).clamp(0.0, 1.0),
    };
    let applies_on = req.applies.unwrap_or(settings.applies);
    let env_num = |name: &str, default: usize| std::env::var(name).ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(default);
    let passages_on = req.passages.unwrap_or_else(|| std::env::var("TIDEMARK_PASSAGES").is_ok_and(|v| v.trim().eq_ignore_ascii_case("on")));
    let passage_opts = passages::Options {
        max_chars: req.passage_max_chars.unwrap_or_else(|| env_num("TIDEMARK_PASSAGE_MAX_CHARS", 400)).max(40),
        min_words: req.passage_min_words.unwrap_or_else(|| env_num("TIDEMARK_PASSAGE_MIN_WORDS", 120)),
    };
    let rerank_candidates = req.rerank_candidates.unwrap_or(settings.rerank_candidates);
    let rerank_all = req.rerank_scope.as_deref().map_or(settings.rerank_all, |v| v.trim().eq_ignore_ascii_case("all"));
    // Ranking is CPU work on the user's records, done in place without copying them.
    let mut data = tokio::task::block_in_place(|| {
        let mut guard = user.lock().unwrap();
        let user = &mut *guard;
        // With the memory engine, its ranking comes first; the hybrid ranking
        // fills the remaining slots, if any: the engine's hits can fill all of
        // top_k, and then the hybrid ranking has no influence on the results.
        // TIDEMARK_LOG_TIMING=1 logs how long each step of a search takes.
        let timing = std::env::var_os("TIDEMARK_LOG_TIMING").is_some();
        let mut marks: Vec<(&str, std::time::Instant)> = vec![("start", std::time::Instant::now())];
        // TIDEMARK_ENGINE_MERGE=fused (per request `engine_merge`): see the merge below. Fusion needs the engine's
        // ranking beyond `top_k` (at least 100), or records it ranks lower get no engine score at all.
        let merge_fused = req.engine_merge.clone().or_else(|| std::env::var("TIDEMARK_ENGINE_MERGE").ok()).is_some_and(|v| v.trim().eq_ignore_ascii_case("fused"));
        let engine_depth = if merge_fused { req.top_k.max(rerank_candidates).max(100) } else { req.top_k.max(rerank_candidates) };
        let engine_first: Vec<String> = match &mut user.engine {
            Some(engine) => {
                let hits = engine.engine.search(&req.user_id, &query, engine_depth.saturating_mul(3));
                let mut seen = std::collections::HashSet::new();
                hits.into_iter()
                    .filter_map(|hit| engine.records.get(&hit.id).map(|(id, _)| id.clone()))
                    .filter(|id| seen.insert(id.clone()))
                    .take(engine_depth)
                    .collect()
            }
            None => Vec::new(),
        };
        let records = &user.records;
        if records.is_empty() {
            return Vec::new();
        }
        marks.push(("engine", std::time::Instant::now()));
        marks.push(("bm25 index", std::time::Instant::now()));
        let lexical = user.bm25.scores(records.iter().map(|r| r.tokens.as_slice()), &search::tokens(&query));
        marks.push(("bm25 score", std::time::Instant::now()));
        let vector = records.iter().map(|r| embed::cosine(&qv, &r.embedding)).collect::<Vec<_>>();
        marks.push(("cosine", std::time::Instant::now()));
        let mut fused = search::fuse(&lexical, &vector, settings.lexical_weight, settings.rrf_k);
        // Smaller units: the same fusion over units; each record scored by its best unit.
        if use_units {
            if let Some(index) = &user.units {
                fused = units::rank_records(index, &search::tokens(&query), &qv, records.len(), &settings, &fused);
            }
        }
        // The user's own lines ranked on their own: questions are mostly about
        // what the user said, which long replies around it would outweigh.
        let query_tokens = search::tokens(&query);
        if user_weight > 0.0 && records.iter().any(|r| !r.user_tokens.is_empty()) {
            let ulex = user.user_bm25.scores(records.iter().map(|r| r.user_tokens.as_slice()), &query_tokens);
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
        marks.push(("fuse + user lines", std::time::Instant::now()));
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
        // TIDEMARK_ENGINE_MERGE=fused (per request `engine_merge`): instead of the engine's hits first and the hybrid
        // ranking filling the rest, the two rankings are combined by reciprocal rank (TIDEMARK_RRF_K), so a record
        // either ranking places high can reach the results. Default "first" (unchanged).
        if !engine_first.is_empty() && merge_fused {
            let position = records.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect::<HashMap<_, _>>();
            let mut score: HashMap<usize, f32> = HashMap::new();
            for (rank, i) in engine_first.iter().filter_map(|id| position.get(id.as_str()).copied()).enumerate() {
                *score.entry(i).or_default() += 1.0 / (settings.rrf_k + rank as f32 + 1.0);
            }
            for (rank, (i, _)) in fused.iter().enumerate() {
                *score.entry(*i).or_default() += 1.0 / (settings.rrf_k + rank as f32 + 1.0);
            }
            // The hybrid ranking covers every record; ties keep its order (stable sort over it).
            let mut merged = fused.iter().map(|(i, _)| (*i, score.get(i).copied().unwrap_or(0.0))).collect::<Vec<_>>();
            merged.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            fused = merged;
        } else if !engine_first.is_empty() {
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
        marks.push(("instructions + rerank + merge", std::time::Instant::now()));
        if timing {
            let parts = marks.windows(2).map(|w| format!("{} {}ms", w[1].0, (w[1].1 - w[0].1).as_millis())).collect::<Vec<_>>();
            eprintln!("search timing user={} records={}: {}", req.user_id, records.len(), parts.join(", "));
        }
        // Instructions (only when instruction slots are enabled) take the last slots, after the ranked records, so
        // the ranking itself is unchanged; the content budget is shared. At most a tenth of the results are
        // instructions (10 of AML's 100 with TIDEMARK_INSTRUCTION_SLOTS=10, the 1.0.0 default). With
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
        // Evidence assembly: related change records join their anchors in the places of unprotected results.
        let mut related_to: HashMap<String, (String, Vec<String>)> = HashMap::new();
        if assembly_max > 0 {
            let ranked = items.iter().map(|it| it.3).collect::<Vec<_>>();
            let (ordered, groups) = assembly::assemble(records, &ranked, &user.user_bm25, assembly_max, settings.assembly_rare, settings.assembly_terms);
            if !groups.is_empty() {
                let terms_of = groups.iter().map(|g| (g.record, g.terms.clone())).collect::<HashMap<_, _>>();
                let score_of = items.iter().map(|it| (it.3, it.2)).collect::<HashMap<_, _>>();
                let mut last_score = items.first().map_or(1.0, |it| it.2);
                items = ordered
                    .into_iter()
                    .map(|(i, anchor)| {
                        let score = match anchor {
                            Some(a) => score_of.get(&a).copied().unwrap_or(last_score) - 1e-4,
                            None => score_of[&i],
                        };
                        last_score = score;
                        if let Some(a) = anchor {
                            related_to.insert(records[i].id.clone(), (records[a].id.clone(), terms_of.get(&i).cloned().unwrap_or_default()));
                        }
                        (records[i].id.clone(), records[i].content.clone(), score, i)
                    })
                    .collect();
            }
        }
        // Follow-ups: the user's statements from other times, attached to the returned records they relate to.
        // They take no slot and change no rank; the content budget still holds.
        let mut followed: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
        let mut used = items.iter().map(|it| it.1.chars().count()).sum::<usize>() + reserved_chars;
        // When a returned statement applies: the period its own words state, resolved against its record's date.
        let mut periods: HashMap<String, Vec<serde_json::Value>> = HashMap::new();
        if applies_on {
            for item in items.iter_mut() {
                let Some(said) = applies::said_in(&records[item.3].content, records[item.3].created_at.as_deref()) else { continue };
                for (sentence, a) in applies::of_record(&records[item.3].content, said) {
                    let note = a.note();
                    if max_chars > 0 && used + note.chars().count() + 1 > max_chars {
                        break;
                    }
                    used += note.chars().count() + 1;
                    item.1.push('\n');
                    item.1.push_str(&note);
                    periods.entry(item.0.clone()).or_default().push(a.json(&sentence));
                }
            }
        }
        if followup.max > 0 {
            let ranked = items.iter().map(|it| it.3).collect::<Vec<_>>();
            for f in followups::find(records, &ranked, &query_tokens, &user.user_bm25, followup) {
                let mut line = followups::line(records, &f);
                let stated = if applies_on { applies::said_in(&records[f.record].content, records[f.record].created_at.as_deref()).map(|said| applies::extract(&f.sentence, said)).unwrap_or_default() } else { Vec::new() };
                for a in &stated {
                    line.push(' ');
                    line.push_str(&a.note());
                }
                if max_chars > 0 && used + line.chars().count() + 1 > max_chars {
                    break;
                }
                used += line.chars().count() + 1;
                let Some(item) = items.iter_mut().find(|it| it.3 == f.anchor) else { continue };
                item.1.push('\n');
                item.1.push_str(&line);
                let mut entry = serde_json::json!({
                    "id": records[f.record].id, "created_at": records[f.record].created_at, "text": f.sentence,
                    "relation": if f.later { "later" } else { "other_time" }, "shared_terms": f.terms,
                });
                if !stated.is_empty() {
                    entry["applies"] = serde_json::json!(stated.iter().map(|a| a.json(&f.sentence)).collect::<Vec<_>>());
                }
                followed.entry(item.0.clone()).or_default().push(entry);
            }
        }
        let last = items.last().map_or(1.0, |it| it.2);
        items.extend(instruction_items.into_iter().enumerate().map(|(j, (id, content, i))| (id, content, last - (j + 1) as f32 * 1e-3, i)));
        items
            .into_iter()
            .map(|(id, content, score, i)| {
                let mut item = serde_json::json!({"id": id, "content": content, "score": score});
                if let Some(c) = &records[i].created_at {
                    item["created_at"] = serde_json::json!(c);
                }
                if let Some(f) = followed.get(&id) {
                    item["followups"] = serde_json::json!(f);
                }
                if let Some(p) = periods.get(&id) {
                    item["applies"] = serde_json::json!(p);
                }
                if passages_on && id == records[i].id {
                    let n = user.bm25.len() as f32;
                    let idf = |t: &str| {
                        let df = user.bm25.df(t) as f32;
                        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
                    };
                    if let Some(p) = passages::best(&records[i].content, &query_tokens, idf, passage_opts) {
                        item["passage"] = p.json();
                    }
                }
                if let Some((anchor, terms)) = related_to.get(&id) {
                    item["kind"] = serde_json::json!("related");
                    item["related_to"] = serde_json::json!(anchor);
                    item["related_terms"] = serde_json::json!(terms);
                }
                item
            })
            .collect::<Vec<_>>()
    });
    // The user's most relevant current preferences take the last slots.
    if preference_slots > 0 && limit > 0 {
        let prefs = preferences::load(&state.db.lock().unwrap(), &req.user_id, false).unwrap_or_default();
        let ranked = preferences::rank(&prefs, &search::tokens(&query), &qv, preference_slots.min(limit), &settings);
        let novelty = preferences::asks_for_novelty(&query);
        let labels = req.preference_labels.as_deref().and_then(preferences::Labels::parse).unwrap_or(state.preferences.labels);
        if !ranked.is_empty() {
            data.truncate(limit - ranked.len());
            let last = data.last().and_then(|d| d["score"].as_f64()).unwrap_or(1.0) as f32;
            data.extend(ranked.iter().enumerate().map(|(j, (i, _))| {
                let p = &prefs[*i];
                let mut item = serde_json::json!({"id": p.id, "content": p.render(labels, novelty), "score": last - (j + 1) as f32 * 1e-3, "kind": "preference"});
                if let Some(at) = &p.stated_at {
                    item["created_at"] = serde_json::json!(at);
                }
                item
            }));
        }
    }
    let mut response = serde_json::json!({"data": data});
    // The user's explicit rules, apart from the results (they take no slot); absent when there are none.
    let rules_on = req.rules.unwrap_or_else(|| std::env::var("TIDEMARK_SEARCH_RULES").map_or(true, |v| !v.trim().eq_ignore_ascii_case("off")));
    if rules_on {
        match rules::for_search(&state.db.lock().unwrap(), &req.user_id, req.rule_scope.as_deref()) {
            Ok(active) if !active.is_empty() => response["rules"] = serde_json::json!(active),
            Ok(_) => {}
            Err(e) => eprintln!("search user={} rules not loaded: {e}", req.user_id),
        }
    }
    Json(response).into_response()
}

fn open_db(path: &str) -> rusqlite::Result<Connection> {
    let db = Connection::open(path)?;
    open_schema(&db)?;
    Ok(db)
}

fn open_schema(db: &Connection) -> rusqlite::Result<()> {
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
         INSERT OR IGNORE INTO user_requests (user_id, request_id, session_id) SELECT user_id, request_id, session_id FROM requests;
         CREATE TABLE IF NOT EXISTS record_history (record_id TEXT NOT NULL, user_id TEXT NOT NULL, version INTEGER NOT NULL,
            action TEXT NOT NULL, content TEXT NOT NULL, changed_at TEXT NOT NULL, PRIMARY KEY (user_id, record_id, version));",
    )?;
    preferences::setup(db)?;
    rules::setup(db)?;
    units::setup(db)
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
        embed::Embedder::Local(local) => local.dims(),
        embed::Embedder::Remote(_) if std::env::var_os("TIDEMARK_SKIP_EMBED_CHECK").is_some() => {
            eprintln!("tidemark: WARNING: embedding check skipped, so stored vectors are not checked against the model");
            return Ok(());
        }
        embed::Embedder::Remote(_) => {
            embedder.embed(&["dimension check".to_string()], embed::Kind::Document).await?.first().map(Vec::len).ok_or("embedding returned no vector")?
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
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.first().map(String::as_str) == Some("units") {
        if let Err(e) = units::build(&args[1..]).await {
            eprintln!("tidemark units: {e}");
            std::process::exit(1);
        }
        return;
    }
    if args.first().map(String::as_str) == Some("reembed") {
        if let Err(e) = reembed::run(&args[1..]).await {
            eprintln!("tidemark reembed: {e}");
            std::process::exit(1);
        }
        return;
    }
    let db_path = std::env::var("TIDEMARK_DB").unwrap_or_else(|_| "tidemark.sqlite".into());
    let port = std::env::var("TIDEMARK_PORT").unwrap_or_else(|_| "18200".into());
    let host = std::env::var("TIDEMARK_HOST").unwrap_or_else(|_| "127.0.0.1".into());
    let db = open_db(&db_path).expect("open database");
    let (users, records): (i64, i64) = db
        .query_row("SELECT COUNT(DISTINCT user_id), COUNT(*) FROM records", [], |r| Ok((r.get(0)?, r.get(1)?)))
        .expect("count records");
    let keys = std::env::var("TIDEMARK_API_KEYS").unwrap_or_default().split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect::<Vec<_>>();
    if let Ok(target) = db.query_row("SELECT value FROM meta WHERE key = ?1", [reembed::IN_PROGRESS], |r| r.get::<_, String>(0)) {
        eprintln!("tidemark: a re-embed to {target} is unfinished; complete it with `tidemark reembed` or drop it with `tidemark reembed --abort`; not starting");
        std::process::exit(1);
    }
    let embedder = embed::Embedder::from_env();
    // A wrong key or URL would fail every Add and Search; refuse to start instead
    // (TIDEMARK_SKIP_EMBED_CHECK=1 skips this, e.g. for offline tests).
    if embedder.is_remote() && std::env::var_os("TIDEMARK_SKIP_EMBED_CHECK").is_none() {
        match embedder.embed(&["startup check".to_string()], embed::Kind::Query).await {
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
    let reranker = match rerank::Reranker::from_env() {
        Ok(r) => r.map(Arc::new),
        Err(e) => {
            eprintln!("tidemark: reranker: {e}; not starting");
            std::process::exit(1);
        }
    };
    // As with embeddings, a wrong reranking key or URL is found at start
    // (TIDEMARK_SKIP_RERANK_CHECK=1 skips this).
    if let Some(rerank::Reranker::Remote(remote)) = reranker.as_deref() {
        if std::env::var_os("TIDEMARK_SKIP_RERANK_CHECK").is_none() {
            match remote.scores("startup check", &["startup check"]).await {
                Ok(_) => eprintln!("tidemark: rerank check ok"),
                Err(e) => {
                    eprintln!("tidemark: rerank check failed, not starting: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    let llm = match llm::from_env().await {
        Ok(m) => m,
        Err(e) => {
            eprintln!("tidemark: {e}; not starting");
            std::process::exit(1);
        }
    };
    let preference_config = match preferences::Config::from_env() {
        Ok(c) if c.mode == preferences::Mode::Llm && llm.is_none() => {
            eprintln!("tidemark: TIDEMARK_PREFERENCES=llm needs a language model (TIDEMARK_LLM_*); not starting");
            std::process::exit(1);
        }
        Ok(c) => c,
        Err(e) => {
            eprintln!("tidemark: {e}; not starting");
            std::process::exit(1);
        }
    };
    if reranker.is_some() && Settings::from_env().rerank_candidates == 0 {
        eprintln!("tidemark: a reranker is configured but TIDEMARK_RERANK_CANDIDATES is 0: it is used only by searches that set rerank_candidates");
    }
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
        "tidemark: {users} users, {records} records in {db_path} (loaded per user on demand, cache {cache_mb} MB); backend {}; embeddings {}; rerank {}; language model {}; preferences {}; auth {}",
        if use_engine { "engine" } else { "builtin" },
        embedder.label(),
        reranker.as_ref().map(|r| r.label()).unwrap_or_else(|| "off".into()),
        llm.as_ref().map(|m| m.label()).unwrap_or_else(|| "off".into()),
        format!("{:?} ({} slots)", preference_config.mode, preference_config.slots).to_lowercase(),
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
        llm,
        preferences: preference_config,
        preference_busy: Mutex::new(Default::default()),
        preference_wake: tokio::sync::Notify::new(),
        keys,
        settings: Settings::from_env(),
    });
    if state.preferences.mode == preferences::Mode::Llm {
        preferences::spawn(state.clone());
    }
    let app = Router::new()
        .route("/health", get(|| async { Json(serde_json::json!({"status": "ok", "service": "tidemark", "version": env!("CARGO_PKG_VERSION")})) }))
        .route("/add", post(add))
        .route("/search", post(search_handler))
        .route("/memories", get(api::list))
        .route("/memories/:id", get(api::get_one).patch(api::update).delete(api::delete_one))
        .route("/memories/:id/history", get(api::history))
        .route("/users/:user_id", axum::routing::delete(api::delete_user))
        .route("/preferences", get(api::list_preferences).post(api::create_preference))
        .route("/preferences/:id", get(api::get_preference).patch(api::update_preference).delete(api::delete_preference))
        .route("/preferences/:id/history", get(api::preference_history))
        .route("/rules", get(rules::list_rules).post(rules::create_rule))
        .route("/rules/:id", get(rules::get_rule).patch(rules::update_rule).delete(rules::delete_rule))
        .route("/rules/:id/history", get(rules::rule_history))
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
    fn instruction_slots_are_off_unless_configured() {
        // 1.2 default (1.0.0 used 10). Only checked when the environment does not set it.
        if std::env::var_os("TIDEMARK_INSTRUCTION_SLOTS").is_none() {
            assert_eq!(Settings::from_env().instruction_slots, 0);
        }
    }

    #[test]
    fn a_record_carries_its_date_speaker_time_notes_and_reply_context() {
        let s = Settings { max_words: 300, previous_context_chars: 20, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 1, rerank_candidates: 0, rerank_all: false, assembly_max: 0, assembly_rare: 0.15, assembly_terms: 8, units: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false, followups: followups::Options { max: 0, rare: 0.05, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 }, applies: false };
        // 8 May 2023 13:56 UTC
        let ts = Some(1683554160000);
        let out = render("user", "Caroline: I went to a support group yesterday.", ts, Some("Melanie: How have you been lately, Caroline?"), &s);
        assert_eq!(out[0].0, "[Monday 8 May 2023, 13:56] Caroline: I went to a support group yesterday. [time: yesterday = 7 May 2023] (in reply to: Melanie: How have yo…)");
        let plain = render("assistant", "Sure, here it is.", None, None, &s);
        assert_eq!(plain[0].0, "assistant: Sure, here it is.");
    }

    #[test]
    fn an_offset_gives_the_senders_calendar_day_and_keeps_the_instant() {
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, assembly_max: 0, assembly_rare: 0.15, assembly_terms: 8, units: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false, followups: followups::Options { max: 0, rare: 0.05, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 }, applies: false };
        // 2025-12-31T22:30:00Z is 00:30 on 1 January 2026 at UTC+02:00.
        let ts = Some(1767220200000);
        let utc = render_at("user", "I arrived yesterday.", ts, None, None, &s);
        assert_eq!(utc[0].0, "[Wednesday 31 December 2025, 22:30] user: I arrived yesterday. [time: yesterday = 30 December 2025]", "without an offset: UTC, as before");
        assert_eq!(utc, render("user", "I arrived yesterday.", ts, None, &s));
        let local = render_at("user", "I arrived yesterday.", ts, Some(120), None, &s);
        assert_eq!(local[0].0, "[Thursday 1 January 2026, 00:30 UTC+02:00] user: I arrived yesterday. [time: yesterday = 31 December 2025]");
        assert_eq!(local[0].1, utc[0].1, "created_at stays the UTC instant");
        assert_eq!(local[0].1.as_deref(), Some("2025-12-31T22:30:00Z"));
        // 2027-01-01T04:30:00Z is 23:30 on 31 December 2026 at UTC−05:00.
        let west = render_at("user", "I'll arrive tomorrow.", Some(1798777800000), Some(-300), None, &s);
        assert_eq!(west[0].0, "[Thursday 31 December 2026, 23:30 UTC-05:00] user: I'll arrive tomorrow. [time: tomorrow = 1 January 2027]");
        assert_eq!(render_at("user", "Hi.", ts, Some(0), None, &s)[0].0, "[Wednesday 31 December 2025, 22:30 UTC+00:00] user: Hi.", "an explicit zero offset is stated");
        assert_eq!(render_at("user", "Hi.", ts, Some(345), None, &s)[0].0, "[Thursday 1 January 2026, 04:15 UTC+05:45] user: Hi.");
        assert!(valid_offset(840) && valid_offset(-720) && !valid_offset(841) && !valid_offset(-721));
        // Mixed offsets in one request: each message's notes use its own offset, never a neighbour's or the header's.
        let m3 = |t: &str, ts: i64| ("user".to_string(), t.to_string(), Some(ts));
        // One instant, 2025-12-31T22:30Z, seen from UTC+02:00 (1 January), from no offset (31 December) and from UTC−05:00 (31 December).
        let mixed = [m3("I arrived yesterday.", 1767220200000), m3("I arrived yesterday.", 1767220200000), m3("I arrived yesterday.", 1767220200000), m3("I leave tomorrow.", 1767220200000)];
        let blocks = render_blocks_at(&mixed, &[Some(120), None, Some(-300), Some(-300)], &s);
        assert_eq!(blocks.iter().map(|b| b.1.as_str()).collect::<Vec<_>>(), [
            "[Thursday 1 January 2026, 00:30 UTC+02:00] \nuser: I arrived yesterday. [time: yesterday = 31 December 2025]",
            "[Wednesday 31 December 2025, 22:30] \nuser: I arrived yesterday. [time: yesterday = 30 December 2025]",
            "[Wednesday 31 December 2025, 17:30 UTC-05:00] \nuser: I arrived yesterday. [time: yesterday = 30 December 2025]\nuser: I leave tomorrow. [time: tomorrow = 1 January 2026]",
        ]);
        assert!(blocks.iter().all(|b| b.2.as_deref() == Some("2025-12-31T22:30:00Z")), "one instant throughout");
        // Search-time applies reads the same day as the Add-time note of the same record.
        for (offset, day) in [(Some(120), "2026-01-02"), (None, "2026-01-01"), (Some(-300), "2026-01-01")] {
            let record = render_blocks_at(&[m3("From tomorrow I'm off caffeine.", 1767220200000)], &[offset], &s).remove(0);
            let said = applies::said_in(&record.1, record.2.as_deref()).unwrap();
            let found = applies::of_record(&record.1, said);
            assert_eq!(found.len(), 1, "{record:?}");
            assert_eq!(found[0].1.from.unwrap().iso(), day);
            let note = chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d").unwrap().format("tomorrow = %-d %B %Y").to_string();
            assert!(record.1.contains(&note), "{} should carry {note}", record.1);
        }
        let single = render_at("user", "From tomorrow I'm off caffeine.", ts, Some(120), None, &s).remove(0);
        assert_eq!(applies::of_record(&single.0, applies::said_in(&single.0, single.1.as_deref()).unwrap())[0].1.from.unwrap().iso(), "2026-01-02", "one message per record");
        // Blocks: the day is the sender's; a block has one header, so another local day or another offset starts a new one.
        let m = |t: &str, ts: i64| ("user".to_string(), t.to_string(), Some(ts));
        let msgs = [m("I arrived yesterday.", 1767220200000), m("Still here.", 1767223800000)];
        let same = render_blocks_at(&msgs, &[Some(120), Some(120)], &s);
        assert_eq!(same.len(), 1);
        assert_eq!(same[0].1, "[Thursday 1 January 2026, 00:30 UTC+02:00] \nuser: I arrived yesterday. [time: yesterday = 31 December 2025]\nuser: Still here.");
        assert_eq!(render_blocks_at(&msgs, &[Some(120), Some(60)], &s).len(), 2, "a different offset is a different header");
        assert_eq!(render_blocks_at(&msgs, &[], &s), render_blocks(&msgs, &s));
        // 23:30 and 00:30 UTC are two UTC days but one day at UTC−05:00.
        let across = [m("One.", 1767223800000), m("Two.", 1767227400000)];
        assert_eq!(render_blocks(&across, &s).len(), 2);
        assert_eq!(render_blocks_at(&across, &[Some(-300), Some(-300)], &s).len(), 1);
    }

    #[test]
    fn messages_are_grouped_into_non_overlapping_dated_blocks() {
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 2, rerank_candidates: 0, rerank_all: false, assembly_max: 0, assembly_rare: 0.15, assembly_terms: 8, units: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false, followups: followups::Options { max: 0, rare: 0.05, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 }, applies: false };
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
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, assembly_max: 0, assembly_rare: 0.15, assembly_terms: 8, units: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false, followups: followups::Options { max: 0, rare: 0.05, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 }, applies: false };
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
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, assembly_max: 0, assembly_rare: 0.15, assembly_terms: 8, units: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false, followups: followups::Options { max: 0, rare: 0.05, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 }, applies: false };
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
        let s = Settings { max_words: 300, previous_context_chars: 0, lexical_weight: 0.5, rrf_k: 60.0, speaker_boost: 0.0, window: 5, rerank_candidates: 0, rerank_all: false, assembly_max: 0, assembly_rare: 0.15, assembly_terms: 8, units: false, max_results: 0, max_chars: 0, user_weight: 0.0, instruction_slots: 0, part_speakers: false, followups: followups::Options { max: 0, rare: 0.05, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 }, applies: false };
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
