//! Memory management beyond Add and Search: list, get, correct and delete
//! memories, delete a user, a memory's history, and the user's preferences.
//!
//! Every request names its `user_id`; a memory is only visible to, and can
//! only be changed by, requests for the user that stored it. Changes run under
//! the user's lock (the one Add and Search take), are durable when the
//! response is sent, and drop the user from the in-memory cache, so the next
//! Search rebuilds the ranking state (and the engine) from SQLite.

use super::*;
use axum::extract::{Path, Query};

#[derive(Deserialize)]
pub struct UserQuery {
    user_id: String,
}

#[derive(Deserialize)]
pub struct ListQuery {
    user_id: String,
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default = "default_list_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_list_limit() -> usize {
    100
}

#[derive(Deserialize)]
pub struct UpdateRequest {
    user_id: String,
    /// The memory's new text, stored as given (it replaces the record's text,
    /// including any date line); its date and ID are kept.
    content: String,
}

fn memory_json(row: &rusqlite::Row) -> rusqlite::Result<serde_json::Value> {
    Ok(serde_json::json!({
        "id": row.get::<_, String>(0)?,
        "session_id": row.get::<_, String>(1)?,
        "request_id": row.get::<_, String>(2)?,
        "content": row.get::<_, String>(3)?,
        "created_at": row.get::<_, Option<String>>(4)?,
    }))
}

const MEMORY_COLUMNS: &str = "id, session_id, request_id, content, created_at";

/// Drops a user's in-memory state; the next request reloads it from SQLite.
fn forget_cached(state: &AppState, user_id: &str) {
    state.cache.lock().unwrap().remove(user_id);
}

fn record_history(tx: &rusqlite::Transaction, record_id: &str, user_id: &str, action: &str, content: &str) -> rusqlite::Result<()> {
    let version: i64 = tx.query_row(
        "SELECT COALESCE(MAX(version), 0) + 1 FROM record_history WHERE user_id = ?1 AND record_id = ?2",
        params![user_id, record_id],
        |r| r.get(0),
    )?;
    tx.execute(
        "INSERT INTO record_history (record_id, user_id, version, action, content, changed_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![record_id, user_id, version, action, content, Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)],
    )?;
    Ok(())
}

/// `GET /memories?user_id=…[&session_id=…][&limit=100][&offset=0]`: the
/// user's memories in the order they were stored.
pub async fn list(State(state): State<Arc<AppState>>, headers: HeaderMap, Query(q): Query<ListQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let limit = q.limit.min(1000);
    let db = state.db.lock().unwrap();
    let total: i64 = match &q.session_id {
        Some(s) => db.query_row("SELECT COUNT(*) FROM records WHERE user_id = ?1 AND session_id = ?2", params![q.user_id, s], |r| r.get(0)),
        None => db.query_row("SELECT COUNT(*) FROM records WHERE user_id = ?1", params![q.user_id], |r| r.get(0)),
    }
    .unwrap_or(0);
    let rows = (|| -> rusqlite::Result<Vec<serde_json::Value>> {
        let sql = format!(
            "SELECT {MEMORY_COLUMNS} FROM records WHERE user_id = ?1 {} ORDER BY seq LIMIT ?2 OFFSET ?3",
            if q.session_id.is_some() { "AND session_id = ?4" } else { "" }
        );
        let mut stmt = db.prepare(&sql)?;
        let mapped = match &q.session_id {
            Some(s) => stmt.query_map(params![q.user_id, limit as i64, q.offset as i64, s], memory_json)?.collect(),
            None => stmt.query_map(params![q.user_id, limit as i64, q.offset as i64], memory_json)?.collect(),
        };
        mapped
    })();
    match rows {
        Ok(data) => Json(serde_json::json!({"data": data, "total": total, "limit": limit, "offset": q.offset})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `GET /memories/{id}?user_id=…`
pub async fn get_one(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<UserQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let db = state.db.lock().unwrap();
    match db.query_row(&format!("SELECT {MEMORY_COLUMNS} FROM records WHERE id = ?1 AND user_id = ?2"), params![id, q.user_id], memory_json) {
        Ok(m) => Json(serde_json::json!({"data": m})).into_response(),
        Err(rusqlite::Error::QueryReturnedNoRows) => error(StatusCode::NOT_FOUND, "no such memory for this user"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `PATCH /memories/{id}` with `{"user_id": …, "content": …}`: corrects a
/// memory's text. It is re-embedded; the previous text is kept in its history.
pub async fn update(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, body: Result<Json<UpdateRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
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
    if req.content.trim().is_empty() {
        return error(StatusCode::BAD_REQUEST, "content must not be empty");
    }
    let lock = user_lock(&state, &req.user_id);
    let _guard = lock.lock().await;
    let old: Option<String> = {
        let db = state.db.lock().unwrap();
        db.query_row("SELECT content FROM records WHERE id = ?1 AND user_id = ?2", params![id, req.user_id], |r| r.get(0)).ok()
    };
    let Some(old) = old else {
        return error(StatusCode::NOT_FOUND, "no such memory for this user");
    };
    let vector = match state.embedder.embed(&[req.content.clone()], embed::Kind::Document).await {
        Ok(mut v) => v.remove(0),
        Err(e) => {
            eprintln!("update user={} embedding failed: {e}", req.user_id);
            return error(StatusCode::SERVICE_UNAVAILABLE, e);
        }
    };
    let blob = vector.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    // A corrected memory is the user's own text: the rules read all of it.
    let created_at: Option<String> = {
        let db = state.db.lock().unwrap();
        db.query_row("SELECT created_at FROM records WHERE id = ?1 AND user_id = ?2", params![id, req.user_id], |r| r.get(0)).ok().flatten()
    };
    let rule_preferences = match state.preferences.mode {
        preferences::Mode::Rules => match preferences::rules_for(&state.embedder, &[(id.clone(), req.content.clone(), created_at.clone())]).await {
            Ok(p) => p,
            Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
        },
        _ => Vec::new(),
    };
    let result = {
        let mut db = state.db.lock().unwrap();
        (|| -> rusqlite::Result<serde_json::Value> {
            let tx = db.transaction()?;
            record_history(&tx, &id, &req.user_id, "updated", &old)?;
            tx.execute(
                "UPDATE records SET content = ?1, raw = ?1, embedding = ?2 WHERE id = ?3 AND user_id = ?4",
                params![req.content, blob, id, req.user_id],
            )?;
            // Preferences drawn from the old text no longer stand on it; the
            // corrected text is read again (rules now, the model in the background).
            preferences::retract_record(&tx, &req.user_id, &id)?;
            match state.preferences.mode {
                preferences::Mode::Rules => preferences::store(&tx, &req.user_id, &rule_preferences)?,
                preferences::Mode::Llm => {
                    preferences::enqueue(&tx, &req.user_id, &[(id.clone(), req.content.clone(), created_at.clone())], state.preferences.delay)?;
                    0
                }
                preferences::Mode::Off => 0,
            };
            let m = tx.query_row(&format!("SELECT {MEMORY_COLUMNS} FROM records WHERE id = ?1 AND user_id = ?2"), params![id, req.user_id], memory_json)?;
            tx.commit()?;
            Ok(m)
        })()
    };
    forget_cached(&state, &req.user_id);
    if state.preferences.mode == preferences::Mode::Llm {
        state.preference_wake.notify_waiters();
    }
    match result {
        Ok(m) => Json(serde_json::json!({"data": m})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `DELETE /memories/{id}?user_id=…`: deletes one memory; its last text is
/// kept in its history (deleting the user removes that too).
pub async fn delete_one(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<UserQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let lock = user_lock(&state, &q.user_id);
    let _guard = lock.lock().await;
    let result = {
        let mut db = state.db.lock().unwrap();
        (|| -> rusqlite::Result<bool> {
            let tx = db.transaction()?;
            let old: Option<String> = tx.query_row("SELECT content FROM records WHERE id = ?1 AND user_id = ?2", params![id, q.user_id], |r| r.get(0)).ok();
            let Some(old) = old else { return Ok(false) };
            record_history(&tx, &id, &q.user_id, "deleted", &old)?;
            tx.execute("DELETE FROM records WHERE id = ?1 AND user_id = ?2", params![id, q.user_id])?;
            preferences::retract_record(&tx, &q.user_id, &id)?;
            tx.commit()?;
            Ok(true)
        })()
    };
    forget_cached(&state, &q.user_id);
    match result {
        Ok(true) => Json(serde_json::json!({"success": true, "deleted": id})).into_response(),
        Ok(false) => error(StatusCode::NOT_FOUND, "no such memory for this user"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `DELETE /users/{user_id}`: deletes everything stored for a user (memories,
/// request log, history). Afterwards the user is unknown to the service.
pub async fn delete_user(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(user_id): Path<String>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let lock = user_lock(&state, &user_id);
    let _guard = lock.lock().await;
    let result = {
        let mut db = state.db.lock().unwrap();
        (|| -> rusqlite::Result<usize> {
            let tx = db.transaction()?;
            let n = tx.execute("DELETE FROM records WHERE user_id = ?1", params![user_id])?;
            tx.execute("DELETE FROM user_requests WHERE user_id = ?1", params![user_id])?;
            tx.execute("DELETE FROM requests WHERE user_id = ?1", params![user_id])?;
            tx.execute("DELETE FROM record_history WHERE user_id = ?1", params![user_id])?;
            preferences::delete_user(&tx, &user_id)?;
            rules::delete_user(&tx, &user_id)?;
            tx.commit()?;
            Ok(n)
        })()
    };
    forget_cached(&state, &user_id);
    match result {
        Ok(n) => Json(serde_json::json!({"success": true, "user_id": user_id, "deleted_memories": n})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `GET /memories/{id}/history?user_id=…`: earlier versions of a memory
/// (its text before each correction, and its last text if deleted).
pub async fn history(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<UserQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let db = state.db.lock().unwrap();
    let rows = (|| -> rusqlite::Result<Vec<serde_json::Value>> {
        let mut stmt = db.prepare("SELECT version, action, content, changed_at FROM record_history WHERE user_id = ?1 AND record_id = ?2 ORDER BY version")?;
        let mapped = stmt.query_map(params![q.user_id, id], |r| {
            Ok(serde_json::json!({"version": r.get::<_, i64>(0)?, "action": r.get::<_, String>(1)?, "content": r.get::<_, String>(2)?, "changed_at": r.get::<_, String>(3)?}))
        })?;
        mapped.collect()
    })();
    match rows {
        Ok(data) => Json(serde_json::json!({"data": data})).into_response(),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

// ---- Preferences ---------------------------------------------------------------------

#[derive(Deserialize)]
pub struct PreferenceListQuery {
    user_id: String,
    /// `current` (default) or `all` (also superseded, retracted and deleted).
    #[serde(default)]
    status: Option<String>,
}

/// `GET /preferences?user_id=…[&status=all]`
pub async fn list_preferences(State(state): State<Arc<AppState>>, headers: HeaderMap, Query(q): Query<PreferenceListQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let all = q.status.as_deref().is_some_and(|s| s.eq_ignore_ascii_case("all"));
    let db = state.db.lock().unwrap();
    match preferences::load(&db, &q.user_id, all) {
        Ok(prefs) => {
            let pending: i64 = db.query_row("SELECT COUNT(*) FROM preference_queue WHERE user_id = ?1", [&q.user_id], |r| r.get(0)).unwrap_or(0);
            Json(serde_json::json!({"data": prefs.iter().map(|p| p.json()).collect::<Vec<_>>(), "pending_records": pending})).into_response()
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `GET /preferences/{id}?user_id=…`
pub async fn get_preference(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<UserQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    match preferences::get(&state.db.lock().unwrap(), &q.user_id, &id) {
        Ok(p) => Json(serde_json::json!({"data": p.json()})).into_response(),
        Err(rusqlite::Error::QueryReturnedNoRows) => error(StatusCode::NOT_FOUND, "no such preference for this user"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

#[derive(Deserialize)]
pub struct PreferenceRequest {
    user_id: String,
    #[serde(default)]
    statement: Option<String>,
    #[serde(default)]
    when_to_apply: Option<String>,
    #[serde(default)]
    category: Option<String>,
    /// `likes`, `does` or `instruction`.
    #[serde(default)]
    kind: Option<String>,
}

/// Embeds and stores one preference given by the caller (`source: "manual"`).
async fn store_manual(state: &AppState, user_id: &str, candidate: preferences::Candidate) -> axum::response::Response {
    let items = match preferences::embedded(&state.embedder, vec![candidate]).await {
        Ok(items) => items,
        Err(e) => return error(StatusCode::SERVICE_UNAVAILABLE, e),
    };
    let mut db = state.db.lock().unwrap();
    let result = (|| -> rusqlite::Result<usize> {
        let tx = db.transaction()?;
        let n = preferences::store(&tx, user_id, &items)?;
        tx.commit()?;
        Ok(n)
    })();
    match result {
        Ok(0) => error(StatusCode::CONFLICT, "the user already has this preference"),
        Ok(_) => {
            let last = preferences::load(&db, user_id, false).ok().and_then(|p| p.into_iter().last());
            Json(serde_json::json!({"data": last.map(|p| p.json())})).into_response()
        }
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `POST /preferences` with `{"user_id", "statement", "when_to_apply"?, "category"?}`:
/// a preference the user or the application states directly.
pub async fn create_preference(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Result<Json<PreferenceRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let Some(statement) = req.statement.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "statement must not be empty");
    };
    let candidate = preferences::Candidate {
        record_id: None,
        statement,
        when_to_apply: req.when_to_apply.filter(|w| !w.trim().is_empty()),
        category: req.category.filter(|c| !c.trim().is_empty()),
        evidence: None,
        stated_at: Some(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        source: "manual",
        replaces: None,
        kind: req.kind.as_deref().and_then(preferences::normalize_kind),
    };
    store_manual(&state, &req.user_id, candidate).await
}

/// `PATCH /preferences/{id}` with `{"user_id", "statement"?, "when_to_apply"?, "category"?}`:
/// a new version of a current preference; the old one is kept as superseded.
pub async fn update_preference(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, body: Result<Json<PreferenceRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let Json(req) = match body {
        Ok(b) => b,
        Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text()),
    };
    let old = match preferences::get(&state.db.lock().unwrap(), &req.user_id, &id) {
        Ok(p) if p.status == "current" => p,
        Ok(_) => return error(StatusCode::CONFLICT, "only a current preference can be changed"),
        Err(rusqlite::Error::QueryReturnedNoRows) => return error(StatusCode::NOT_FOUND, "no such preference for this user"),
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let statement = req.statement.map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).unwrap_or(old.statement.clone());
    let candidate = preferences::Candidate {
        // The user's correction is the authority now, not the old quote.
        record_id: None,
        evidence: None,
        when_to_apply: req.when_to_apply.map_or(old.when_to_apply.clone(), |w| Some(w).filter(|w| !w.trim().is_empty())),
        category: req.category.map_or(old.category.clone(), |c| Some(c).filter(|c| !c.trim().is_empty())),
        statement,
        stated_at: Some(chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)),
        source: "manual",
        replaces: Some(id),
        kind: req.kind.as_deref().and_then(preferences::normalize_kind).or(old.kind.clone()),
    };
    store_manual(&state, &req.user_id, candidate).await
}

/// `DELETE /preferences/{id}?user_id=…`: the preference no longer applies (kept in its history).
pub async fn delete_preference(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<UserQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let db = state.db.lock().unwrap();
    match db.execute(
        "UPDATE preferences SET status = 'deleted', updated_at = ?1 WHERE id = ?2 AND user_id = ?3 AND status = 'current'",
        params![Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true), id, q.user_id],
    ) {
        Ok(1) => Json(serde_json::json!({"success": true, "deleted": id})).into_response(),
        Ok(_) => error(StatusCode::NOT_FOUND, "no such current preference for this user"),
        Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    }
}

/// `GET /preferences/{id}/history?user_id=…`: every version of the
/// preference, oldest first (the versions it replaced and those replacing it).
pub async fn preference_history(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<UserQuery>) -> axum::response::Response {
    if !authorized(&state, &headers) {
        return error(StatusCode::UNAUTHORIZED, "unauthorized");
    }
    let db = state.db.lock().unwrap();
    let all = match preferences::load(&db, &q.user_id, true) {
        Ok(a) => a,
        Err(e) => return error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
    };
    let Some(start) = all.iter().find(|p| p.id == id) else {
        return error(StatusCode::NOT_FOUND, "no such preference for this user");
    };
    let mut chain = vec![start.clone()];
    while let Some(prev) = chain[0].replaces.as_ref().and_then(|r| all.iter().find(|p| &p.id == r)) {
        chain.insert(0, prev.clone());
    }
    while let Some(next) = all.iter().find(|p| p.replaces.as_ref() == Some(&chain[chain.len() - 1].id)) {
        chain.push(next.clone());
    }
    Json(serde_json::json!({"data": chain.iter().map(|p| p.json()).collect::<Vec<_>>()})).into_response()
}
