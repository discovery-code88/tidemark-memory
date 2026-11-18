//! Explicit standing rules: what the user tells Tidemark to follow ("Don't use Git"), stated on purpose through the
//! API rather than recognised in conversation text. Nothing in Add or Search creates, changes or revokes a rule; only
//! these endpoints do.
//!
//! - `POST /rules` `{"user_id", "text", "scope"?}`: adds an active rule (an identical active rule is returned as is).
//! - `GET /rules?user_id=…[&status=active|revoked|all]`: the user's rules (default: active).
//! - `GET /rules/{id}?user_id=…`, `GET /rules/{id}/history?user_id=…`: one rule; all its versions.
//! - `PATCH /rules/{id}` `{"user_id", "text"?, "scope"?}`: a new version of an active rule (`"scope": ""` clears it).
//! - `DELETE /rules/{id}?user_id=…`: revokes the rule (kept, with its history); `&purge=true` erases it entirely.
//! - Search returns the user's active rules in a separate `rules` field, next to `data`; they never take a result
//!   slot. Per request `rules: false` leaves them out and `rule_scope` limits them to that scope plus unscoped rules.
//!   The field is absent when the user has no active rules, so responses for users without rules are unchanged.
//!
//! Versioning follows the preference store: every change is a new version and nothing is overwritten. Rules that may
//! contradict each other are never resolved automatically: all stay active and each names the others under
//! `possible_conflicts` (same or unscoped scope, a shared content word, opposite negation). That check only points;
//! it decides nothing, and it is incomplete: contradictions without a shared word are not found, so an empty list
//! does not mean a rule conflicts with nothing.

use super::*;
use axum::extract::{Path, Query};

pub fn setup(db: &Connection) -> rusqlite::Result<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS rules (rule_id TEXT NOT NULL, user_id TEXT NOT NULL, version INTEGER NOT NULL, seq INTEGER NOT NULL,
            text TEXT NOT NULL, scope TEXT, status TEXT NOT NULL, created_at TEXT NOT NULL, changed_at TEXT NOT NULL,
            PRIMARY KEY (user_id, rule_id, version));
         CREATE INDEX IF NOT EXISTS rules_user ON rules(user_id, seq);",
    )
}

/// One version of a rule.
#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub id: String,
    pub version: i64,
    pub text: String,
    pub scope: Option<String>,
    pub status: String,
    pub created_at: String,
    pub changed_at: String,
}

impl Rule {
    pub fn json(&self, conflicts: &[String]) -> serde_json::Value {
        serde_json::json!({
            "id": self.id, "text": self.text, "scope": self.scope, "status": self.status, "version": self.version,
            "created_at": self.created_at, "updated_at": self.changed_at, "possible_conflicts": conflicts,
        })
    }
}

fn now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

fn clean(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn clean_scope(scope: Option<&str>) -> Option<String> {
    scope.map(clean).filter(|s| !s.is_empty())
}

const COLUMNS: &str = "rule_id, version, text, scope, status, created_at, changed_at";

fn row(r: &rusqlite::Row) -> rusqlite::Result<Rule> {
    Ok(Rule { id: r.get(0)?, version: r.get(1)?, text: r.get(2)?, scope: r.get(3)?, status: r.get(4)?, created_at: r.get(5)?, changed_at: r.get(6)? })
}

/// The latest version of each of the user's rules, in the order they were added; `status` filters (None = all).
pub fn latest(db: &Connection, user_id: &str, status: Option<&str>) -> rusqlite::Result<Vec<Rule>> {
    let sql = format!(
        "SELECT {COLUMNS} FROM rules r WHERE user_id = ?1 AND version = (SELECT MAX(version) FROM rules WHERE user_id = r.user_id AND rule_id = r.rule_id) ORDER BY seq"
    );
    let mut stmt = db.prepare(&sql)?;
    let rows = stmt.query_map([user_id], row)?.collect::<rusqlite::Result<Vec<_>>>()?;
    Ok(rows.into_iter().filter(|r| status.is_none_or(|s| r.status == s)).collect())
}

pub fn get(db: &Connection, user_id: &str, id: &str) -> rusqlite::Result<Option<Rule>> {
    let mut stmt = db.prepare(&format!("SELECT {COLUMNS} FROM rules WHERE user_id = ?1 AND rule_id = ?2 ORDER BY version DESC LIMIT 1"))?;
    let mut rows = stmt.query_map(params![user_id, id], row)?;
    rows.next().transpose()
}

pub fn history(db: &Connection, user_id: &str, id: &str) -> rusqlite::Result<Vec<Rule>> {
    let mut stmt = db.prepare(&format!("SELECT {COLUMNS} FROM rules WHERE user_id = ?1 AND rule_id = ?2 ORDER BY version"))?;
    let rows = stmt.query_map(params![user_id, id], row)?;
    rows.collect()
}

fn insert(tx: &rusqlite::Transaction, user_id: &str, rule: &Rule, seq: i64) -> rusqlite::Result<()> {
    tx.execute(
        "INSERT INTO rules (rule_id, user_id, version, seq, text, scope, status, created_at, changed_at) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![rule.id, user_id, rule.version, seq, rule.text, rule.scope, rule.status, rule.created_at, rule.changed_at],
    )?;
    Ok(())
}

fn seq_of(tx: &rusqlite::Transaction, user_id: &str, id: &str) -> rusqlite::Result<i64> {
    tx.query_row("SELECT seq FROM rules WHERE user_id = ?1 AND rule_id = ?2 AND version = 1", params![user_id, id], |r| r.get(0))
}

/// Adds an active rule; an active rule with the same text and scope is returned instead (`false` = not new).
pub fn create(tx: &rusqlite::Transaction, user_id: &str, text: &str, scope: Option<&str>) -> rusqlite::Result<(Rule, bool)> {
    let (text, scope) = (clean(text), clean_scope(scope));
    if let Some(same) = latest(tx, user_id, Some("active"))?.into_iter().find(|r| r.text.to_lowercase() == text.to_lowercase() && r.scope == scope) {
        return Ok((same, false));
    }
    let seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq), 0) + 1 FROM rules", [], |r| r.get(0))?;
    let at = now();
    let rule = Rule { id: format!("rule_{}", short_hash(&format!("{user_id}\u{0}{text}\u{0}{seq}"))), version: 1, text, scope, status: "active".into(), created_at: at.clone(), changed_at: at };
    insert(tx, user_id, &rule, seq)?;
    Ok((rule, true))
}

pub enum Change {
    Done(Rule),
    NotFound,
    NotActive,
}

/// A new version of an active rule with a new text and/or scope (`Some("")` clears the scope).
pub fn update(tx: &rusqlite::Transaction, user_id: &str, id: &str, text: Option<&str>, scope: Option<&str>) -> rusqlite::Result<Change> {
    let Some(old) = get(tx, user_id, id)? else { return Ok(Change::NotFound) };
    if old.status != "active" {
        return Ok(Change::NotActive);
    }
    let rule = Rule {
        version: old.version + 1,
        text: text.map(clean).filter(|t| !t.is_empty()).unwrap_or(old.text.clone()),
        scope: match scope { Some(s) => clean_scope(Some(s)), None => old.scope.clone() },
        changed_at: now(),
        ..old
    };
    insert(tx, user_id, &rule, seq_of(tx, user_id, id)?)?;
    Ok(Change::Done(rule))
}

/// Revokes an active rule (a new version with status `revoked`; earlier versions stay).
pub fn revoke(tx: &rusqlite::Transaction, user_id: &str, id: &str) -> rusqlite::Result<Change> {
    let Some(old) = get(tx, user_id, id)? else { return Ok(Change::NotFound) };
    if old.status != "active" {
        return Ok(Change::NotActive);
    }
    let rule = Rule { version: old.version + 1, status: "revoked".into(), changed_at: now(), ..old };
    insert(tx, user_id, &rule, seq_of(tx, user_id, id)?)?;
    Ok(Change::Done(rule))
}

/// Erases a rule and its history; returns whether it existed.
pub fn purge(tx: &rusqlite::Transaction, user_id: &str, id: &str) -> rusqlite::Result<bool> {
    Ok(tx.execute("DELETE FROM rules WHERE user_id = ?1 AND rule_id = ?2", params![user_id, id])? > 0)
}

pub fn delete_user(tx: &rusqlite::Transaction, user_id: &str) -> rusqlite::Result<()> {
    tx.execute("DELETE FROM rules WHERE user_id = ?1", [user_id])?;
    Ok(())
}

/// Words that make a rule a prohibition (lower case; "don't" also as "dont" and "don t" after tokenizing).
const NEGATIONS: &[&str] = &["don't", "dont", "do not", "never", "no", "not", "avoid", "stop", "without", "mustn't", "must not", "shouldn't", "should not", "can't", "cannot", "won't"];

fn negative(text: &str) -> bool {
    let t = format!(" {} ", text.to_lowercase().replace('’', "'").split(|c: char| !(c.is_alphanumeric() || c == '\'')).filter(|w| !w.is_empty()).collect::<Vec<_>>().join(" "));
    NEGATIONS.iter().any(|n| t.contains(&format!(" {n} ")))
}

fn content_words(text: &str) -> std::collections::HashSet<String> {
    let neg = ["dont", "don", "never", "avoid", "stop", "without", "mustn", "shouldn", "cannot", "won", "use", "always", "please", "make", "sure"];
    search::tokens(text).into_iter().filter(|t| t.chars().count() > 2 && !neg.contains(&t.as_str())).collect()
}

/// For each rule, the other rules it may contradict: scopes overlap (equal, or either unscoped), they share a content
/// word, and exactly one is a prohibition. A pointer for the reader; nothing is resolved.
pub fn possible_conflicts(rules: &[Rule]) -> Vec<Vec<String>> {
    let words = rules.iter().map(|r| content_words(&r.text)).collect::<Vec<_>>();
    let neg = rules.iter().map(|r| negative(&r.text)).collect::<Vec<_>>();
    let scope = |r: &Rule| r.scope.as_ref().map(|s| s.to_lowercase());
    (0..rules.len())
        .map(|i| {
            (0..rules.len())
                .filter(|&j| j != i)
                .filter(|&j| {
                    let overlap = scope(&rules[i]).is_none() || scope(&rules[j]).is_none() || scope(&rules[i]) == scope(&rules[j]);
                    overlap && neg[i] != neg[j] && !words[i].is_disjoint(&words[j])
                })
                .map(|j| rules[j].id.clone())
                .collect()
        })
        .collect()
}

/// The user's active rules as returned by Search (with possible conflicts among them), limited to `scope` plus
/// unscoped rules when a scope is given.
pub fn for_search(db: &Connection, user_id: &str, scope: Option<&str>) -> rusqlite::Result<Vec<serde_json::Value>> {
    let active = latest(db, user_id, Some("active"))?;
    let conflicts = possible_conflicts(&active);
    let want = scope.map(|s| clean(s).to_lowercase()).filter(|s| !s.is_empty());
    Ok(active
        .iter()
        .zip(&conflicts)
        .filter(|(r, _)| match (&want, &r.scope) {
            (None, _) | (_, None) => true,
            (Some(w), Some(s)) => s.to_lowercase() == *w,
        })
        .map(|(r, c)| r.json(c))
        .collect())
}

// ---- HTTP

#[derive(Deserialize)]
pub struct RuleRequest {
    user_id: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    scope: Option<String>,
}

#[derive(Deserialize)]
pub struct ListQuery {
    user_id: String,
    #[serde(default)]
    status: Option<String>,
}

#[derive(Deserialize)]
pub struct OneQuery {
    user_id: String,
    #[serde(default)]
    purge: Option<bool>,
}

fn with_conflicts(db: &Connection, user_id: &str, rule: &Rule) -> rusqlite::Result<serde_json::Value> {
    let active = latest(db, user_id, Some("active"))?;
    let conflicts = possible_conflicts(&active);
    let c = active.iter().position(|r| r.id == rule.id).map(|i| conflicts[i].clone()).unwrap_or_default();
    Ok(rule.json(&c))
}

macro_rules! guard {
    ($state:expr, $headers:expr) => {
        if !authorized(&$state, &$headers) {
            return error(StatusCode::UNAUTHORIZED, "unauthorized");
        }
    };
}

fn parse<T>(body: Result<Json<T>, axum::extract::rejection::JsonRejection>) -> Result<T, axum::response::Response> {
    body.map(|Json(b)| b).map_err(|e| error(StatusCode::BAD_REQUEST, e.body_text()))
}

fn internal(e: rusqlite::Error) -> axum::response::Response {
    error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

pub async fn create_rule(State(state): State<Arc<AppState>>, headers: HeaderMap, body: Result<Json<RuleRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
    guard!(state, headers);
    let req = match parse(body) { Ok(r) => r, Err(e) => return e };
    let Some(text) = req.text.as_deref().map(clean).filter(|t| !t.is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "text must not be empty");
    };
    let mut db = state.db.lock().unwrap();
    let result = (|| -> rusqlite::Result<(serde_json::Value, bool)> {
        let tx = db.transaction()?;
        let (rule, created) = create(&tx, &req.user_id, &text, req.scope.as_deref())?;
        let json = with_conflicts(&tx, &req.user_id, &rule)?;
        tx.commit()?;
        Ok((json, created))
    })();
    match result {
        Ok((rule, true)) => (StatusCode::CREATED, Json(serde_json::json!({"success": true, "created": true, "rule": rule}))).into_response(),
        Ok((rule, false)) => Json(serde_json::json!({"success": true, "created": false, "rule": rule})).into_response(),
        Err(e) => internal(e),
    }
}

pub async fn list_rules(State(state): State<Arc<AppState>>, headers: HeaderMap, Query(q): Query<ListQuery>) -> axum::response::Response {
    guard!(state, headers);
    let status = match q.status.as_deref().map(str::trim) {
        None | Some("") | Some("active") => Some("active"),
        Some("revoked") => Some("revoked"),
        Some("all") => None,
        Some(other) => return error(StatusCode::BAD_REQUEST, format!("status must be active, revoked or all, not {other:?}")),
    };
    let db = state.db.lock().unwrap();
    let all = match latest(&db, &q.user_id, None) { Ok(r) => r, Err(e) => return internal(e) };
    let active = all.iter().filter(|r| r.status == "active").cloned().collect::<Vec<_>>();
    let conflicts = possible_conflicts(&active);
    let data = all
        .iter()
        .filter(|r| status.is_none_or(|s| r.status == s))
        .map(|r| r.json(&active.iter().position(|a| a.id == r.id).map(|i| conflicts[i].clone()).unwrap_or_default()))
        .collect::<Vec<_>>();
    Json(serde_json::json!({"data": data, "total": data.len()})).into_response()
}

pub async fn get_rule(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<OneQuery>) -> axum::response::Response {
    guard!(state, headers);
    let db = state.db.lock().unwrap();
    match get(&db, &q.user_id, &id) {
        Ok(Some(rule)) => match with_conflicts(&db, &q.user_id, &rule) { Ok(j) => Json(j).into_response(), Err(e) => internal(e) },
        Ok(None) => error(StatusCode::NOT_FOUND, "no such rule for this user"),
        Err(e) => internal(e),
    }
}

pub async fn rule_history(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<OneQuery>) -> axum::response::Response {
    guard!(state, headers);
    let db = state.db.lock().unwrap();
    match history(&db, &q.user_id, &id) {
        Ok(v) if v.is_empty() => error(StatusCode::NOT_FOUND, "no such rule for this user"),
        Ok(v) => Json(serde_json::json!({"id": id, "versions": v.iter().map(|r| r.json(&[])).collect::<Vec<_>>()})).into_response(),
        Err(e) => internal(e),
    }
}

pub async fn update_rule(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, body: Result<Json<RuleRequest>, axum::extract::rejection::JsonRejection>) -> axum::response::Response {
    guard!(state, headers);
    let req = match parse(body) { Ok(r) => r, Err(e) => return e };
    if req.text.as_deref().map(clean).filter(|t| !t.is_empty()).is_none() && req.scope.is_none() {
        return error(StatusCode::BAD_REQUEST, "give a new text and/or scope");
    }
    let mut db = state.db.lock().unwrap();
    let result = (|| -> rusqlite::Result<Result<serde_json::Value, axum::response::Response>> {
        let tx = db.transaction()?;
        let out = match update(&tx, &req.user_id, &id, req.text.as_deref(), req.scope.as_deref())? {
            Change::Done(rule) => Ok(with_conflicts(&tx, &req.user_id, &rule)?),
            Change::NotFound => Err(error(StatusCode::NOT_FOUND, "no such rule for this user")),
            Change::NotActive => Err(error(StatusCode::CONFLICT, "only an active rule can be changed")),
        };
        tx.commit()?;
        Ok(out)
    })();
    match result {
        Ok(Ok(rule)) => Json(serde_json::json!({"success": true, "rule": rule})).into_response(),
        Ok(Err(resp)) => resp,
        Err(e) => internal(e),
    }
}

pub async fn delete_rule(State(state): State<Arc<AppState>>, headers: HeaderMap, Path(id): Path<String>, Query(q): Query<OneQuery>) -> axum::response::Response {
    guard!(state, headers);
    let mut db = state.db.lock().unwrap();
    let result = (|| -> rusqlite::Result<axum::response::Response> {
        let tx = db.transaction()?;
        let resp = if q.purge.unwrap_or(false) {
            if purge(&tx, &q.user_id, &id)? {
                Json(serde_json::json!({"success": true, "id": id, "purged": true})).into_response()
            } else {
                error(StatusCode::NOT_FOUND, "no such rule for this user")
            }
        } else {
            match revoke(&tx, &q.user_id, &id)? {
                Change::Done(rule) => Json(serde_json::json!({"success": true, "rule": rule.json(&[])})).into_response(),
                Change::NotFound => error(StatusCode::NOT_FOUND, "no such rule for this user"),
                Change::NotActive => error(StatusCode::CONFLICT, "the rule is already revoked"),
            }
        };
        tx.commit()?;
        Ok(resp)
    })();
    result.unwrap_or_else(internal)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Connection {
        let db = Connection::open_in_memory().unwrap();
        setup(&db).unwrap();
        db
    }

    #[test]
    fn a_rule_is_added_listed_changed_and_revoked_with_its_history_kept() {
        let mut db = db();
        let tx = db.transaction().unwrap();
        let (r, created) = create(&tx, "u", "  Don't use   Git ", Some("projects")).unwrap();
        assert!(created);
        assert_eq!((r.text.as_str(), r.scope.as_deref(), r.status.as_str(), r.version), ("Don't use Git", Some("projects"), "active", 1));
        // Adding it again returns the same rule.
        let (again, created) = create(&tx, "u", "don't use git", Some("projects")).unwrap();
        assert!(!created);
        assert_eq!(again.id, r.id);
        let Change::Done(v2) = update(&tx, "u", &r.id, Some("Don't use Git; use tarball snapshots"), None).unwrap() else { panic!() };
        assert_eq!((v2.version, v2.scope.as_deref()), (2, Some("projects")));
        let Change::Done(v3) = update(&tx, "u", &r.id, None, Some("")).unwrap() else { panic!() };
        assert_eq!((v3.version, v3.scope), (3, None));
        let Change::Done(gone) = revoke(&tx, "u", &r.id).unwrap() else { panic!() };
        assert_eq!((gone.status.as_str(), gone.version), ("revoked", 4));
        assert!(matches!(revoke(&tx, "u", &r.id).unwrap(), Change::NotActive));
        assert!(matches!(update(&tx, "u", &r.id, Some("x"), None).unwrap(), Change::NotActive));
        assert_eq!(history(&tx, "u", &r.id).unwrap().len(), 4);
        assert!(latest(&tx, "u", Some("active")).unwrap().is_empty());
        assert_eq!(latest(&tx, "u", Some("revoked")).unwrap().len(), 1);
        tx.commit().unwrap();
    }

    #[test]
    fn rules_belong_to_their_user() {
        let mut db = db();
        let tx = db.transaction().unwrap();
        let (r, _) = create(&tx, "alice", "Never print API keys", None).unwrap();
        assert!(get(&tx, "bob", &r.id).unwrap().is_none());
        assert!(matches!(revoke(&tx, "bob", &r.id).unwrap(), Change::NotFound));
        assert!(!purge(&tx, "bob", &r.id).unwrap());
        assert!(latest(&tx, "bob", None).unwrap().is_empty());
        assert!(purge(&tx, "alice", &r.id).unwrap());
        assert!(history(&tx, "alice", &r.id).unwrap().is_empty());
    }

    #[test]
    fn contradicting_rules_both_stay_active_and_point_at_each_other() {
        let mut db = db();
        let tx = db.transaction().unwrap();
        let (a, _) = create(&tx, "u", "Don't use Git", None).unwrap();
        let (b, _) = create(&tx, "u", "Use Git for every snapshot", Some("work")).unwrap();
        let (c, _) = create(&tx, "u", "Never print API keys", None).unwrap();
        let (d, _) = create(&tx, "u", "Write commit messages in English", Some("home")).unwrap();
        let (e, _) = create(&tx, "u", "Don't write messages in German", Some("work")).unwrap();
        let active = latest(&tx, "u", Some("active")).unwrap();
        assert_eq!(active.len(), 5, "nothing is resolved automatically");
        let conflicts = possible_conflicts(&active);
        let of = |id: &str| conflicts[active.iter().position(|r| r.id == id).unwrap()].clone();
        assert_eq!(of(&a.id), vec![b.id.clone()]);
        assert_eq!(of(&b.id), vec![a.id.clone()]);
        assert!(of(&c.id).is_empty());
        // Different scopes (home / work) are not pointed at each other.
        assert!(of(&d.id).is_empty() && of(&e.id).is_empty());
    }

    #[test]
    fn search_returns_rules_of_a_scope_plus_unscoped_ones() {
        let mut db = db();
        let tx = db.transaction().unwrap();
        create(&tx, "u", "Don't use Git", None).unwrap();
        create(&tx, "u", "Answer in French", Some("email")).unwrap();
        let (revoked, _) = create(&tx, "u", "Use metric units", None).unwrap();
        revoke(&tx, "u", &revoked.id).unwrap();
        tx.commit().unwrap();
        assert_eq!(for_search(&db, "u", None).unwrap().len(), 2);
        let code = for_search(&db, "u", Some("code")).unwrap();
        assert_eq!(code.len(), 1);
        assert_eq!(code[0]["text"], "Don't use Git");
        assert_eq!(for_search(&db, "u", Some(" Email ")).unwrap().len(), 2);
        assert!(for_search(&db, "nobody", None).unwrap().is_empty());
    }
}
