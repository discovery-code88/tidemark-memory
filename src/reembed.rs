//! `tidemark reembed`: moves a database to another embedding model (or the same
//! model with other settings) without losing any memory.
//!
//! Run it with the service stopped and the new `TIDEMARK_EMBED_*` settings.
//! New vectors are staged in their own table, chunk by chunk, so an interrupted
//! run resumes where it stopped and the old vectors stay usable until the end.
//! Only when every record has a new vector are they swapped in, together with
//! the database's recorded model, in one transaction.
//!
//! `--estimate` counts records and characters without calling the provider;
//! `--abort` drops staged vectors. While a run is unfinished the service
//! refuses to start, since staged and stored vectors may come from different models.

use super::*;

pub const IN_PROGRESS: &str = "reembed_target";

/// The text embedded for a stored record. Records of blocks (`…#bN`) and
/// corrected records keep it in `raw`; with `TIDEMARK_WINDOW=1` `raw` is the
/// whole message and a split message's part is recomputed (its role is not
/// stored, so a part of a message without a speaker is embedded without one).
fn embed_input(id: &str, raw: &str, s: &Settings) -> String {
    let part = id.rsplit_once('#').and_then(|(_, tail)| tail.split_once('.')).and_then(|(_, p)| p.parse::<usize>().ok());
    match part {
        Some(p) => embed_texts("", raw, s).into_iter().nth(p).unwrap_or_else(|| raw.to_string()),
        None => raw.to_string(),
    }
}

fn setup(db: &Connection) -> rusqlite::Result<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS reembed (id TEXT PRIMARY KEY, embedding BLOB NOT NULL);
         CREATE TABLE IF NOT EXISTS reembed_preferences (id TEXT PRIMARY KEY, embedding BLOB NOT NULL);",
    )
}

fn meta(db: &Connection, key: &str) -> Option<String> {
    db.query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0)).ok()
}

pub async fn run(args: &[String]) -> Result<(), String> {
    let db_path = std::env::var("TIDEMARK_DB").unwrap_or_else(|_| "tidemark.sqlite".into());
    let mut db = open_db(&db_path).map_err(|e| e.to_string())?;
    setup(&db).map_err(|e| e.to_string())?;
    let settings = Settings::from_env();
    let sql = |e: rusqlite::Error| e.to_string();
    if args.iter().any(|a| a == "--abort") {
        let n: i64 = db.query_row("SELECT COUNT(*) FROM reembed", [], |r| r.get(0)).map_err(sql)?;
        db.execute_batch(&format!("DELETE FROM reembed; DELETE FROM reembed_preferences; DELETE FROM meta WHERE key = '{IN_PROGRESS}';")).map_err(sql)?;
        eprintln!("tidemark reembed: aborted; {n} staged vectors dropped, stored vectors unchanged");
        return Ok(());
    }
    let embedder = embed::Embedder::from_env();
    let target = embedder.label();
    let (inputs_total, chars): (i64, i64) = db
        .query_row(
            "SELECT (SELECT COUNT(*) FROM records) + (SELECT COUNT(*) FROM preferences),
                    (SELECT COALESCE(SUM(length(raw)), 0) FROM records) + (SELECT COALESCE(SUM(length(statement) + COALESCE(length(when_to_apply), 0)), 0) FROM preferences)",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .map_err(sql)?;
    let records: i64 = db.query_row("SELECT COUNT(*) FROM records", [], |r| r.get(0)).map_err(sql)?;
    let current = meta(&db, "embedding_model").unwrap_or_else(|| "unrecorded".into());
    if args.iter().any(|a| a == "--estimate") {
        eprintln!(
            "tidemark reembed: {inputs_total} records and preferences, {chars} characters of input (capped per record by TIDEMARK_EMBED_MAX_CHARS); \
             stored {current}, target {target}. Tokens are roughly characters/4 for English, up to characters for Chinese."
        );
        return Ok(());
    }
    if let Some(previous) = meta(&db, IN_PROGRESS) {
        if previous != target {
            let n = db.execute("DELETE FROM reembed", []).map_err(sql)?;
            db.execute("DELETE FROM reembed_preferences", []).map_err(sql)?;
            eprintln!("tidemark reembed: the unfinished run targeted {previous}; its {n} staged vectors are dropped");
        }
    }
    let dims = embedder.embed(&["dimension check".to_string()], embed::Kind::Document).await?.first().map(Vec::len).ok_or("embedding returned no vector")?;
    db.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![IN_PROGRESS, target]).map_err(sql)?;
    let chunk = std::env::var("TIDEMARK_REEMBED_CHUNK").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(200).max(1);
    eprintln!("tidemark reembed: {current} -> {target} ({dims} dimensions), {records} records in {db_path}");
    let started = std::time::Instant::now();
    loop {
        let todo = {
            let mut stmt = db
                .prepare("SELECT id, raw FROM records WHERE id NOT IN (SELECT id FROM reembed) ORDER BY seq LIMIT ?1")
                .map_err(sql)?;
            let rows = stmt.query_map([chunk as i64], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))).map_err(sql)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(sql)?
        };
        if todo.is_empty() {
            break;
        }
        let inputs = todo.iter().map(|(id, raw)| embed_input(id, raw, &settings)).collect::<Vec<_>>();
        let vectors = embedder.embed(&inputs, embed::Kind::Document).await?;
        if let Some(v) = vectors.iter().find(|v| v.len() != dims) {
            return Err(format!("the provider returned a {}-dimension vector, expected {dims}; staged vectors are kept", v.len()));
        }
        let tx = db.transaction().map_err(sql)?;
        for ((id, _), v) in todo.iter().zip(&vectors) {
            let blob = v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            tx.execute("INSERT OR REPLACE INTO reembed (id, embedding) VALUES (?1, ?2)", params![id, blob]).map_err(sql)?;
        }
        tx.commit().map_err(sql)?;
        let done: i64 = db.query_row("SELECT COUNT(*) FROM reembed", [], |r| r.get(0)).map_err(sql)?;
        eprintln!(
            "tidemark reembed: {done}/{records} records, {} tokens reported, {:.0} s",
            embed::TOKENS_USED.load(std::sync::atomic::Ordering::Relaxed),
            started.elapsed().as_secs_f64()
        );
    }
    // Preferences (all versions) are embedded by statement and situations.
    loop {
        let todo = {
            let mut stmt = db
                .prepare("SELECT id, statement, when_to_apply FROM preferences WHERE id NOT IN (SELECT id FROM reembed_preferences) ORDER BY seq LIMIT ?1")
                .map_err(sql)?;
            let rows = stmt.query_map([chunk as i64], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Option<String>>(2)?))).map_err(sql)?;
            rows.collect::<Result<Vec<_>, _>>().map_err(sql)?
        };
        if todo.is_empty() {
            break;
        }
        let inputs = todo
            .iter()
            .map(|(_, statement, when)| preferences::Candidate { record_id: None, statement: statement.clone(), when_to_apply: when.clone(), category: None, evidence: None, stated_at: None, source: "", replaces: None, kind: None }.index_text())
            .collect::<Vec<_>>();
        let vectors = embedder.embed(&inputs, embed::Kind::Document).await?;
        let tx = db.transaction().map_err(sql)?;
        for ((id, _, _), v) in todo.iter().zip(&vectors) {
            let blob = v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
            tx.execute("INSERT OR REPLACE INTO reembed_preferences (id, embedding) VALUES (?1, ?2)", params![id, blob]).map_err(sql)?;
        }
        tx.commit().map_err(sql)?;
        eprintln!("tidemark reembed: {} preferences staged", db.query_row("SELECT COUNT(*) FROM reembed_preferences", [], |r| r.get::<_, i64>(0)).map_err(sql)?);
    }
    let tx = db.transaction().map_err(sql)?;
    let missing: i64 = tx.query_row("SELECT COUNT(*) FROM records WHERE id NOT IN (SELECT id FROM reembed)", [], |r| r.get(0)).map_err(sql)?;
    if missing > 0 {
        return Err(format!("{missing} records have no new vector; run again"));
    }
    tx.execute("UPDATE records SET embedding = (SELECT embedding FROM reembed WHERE reembed.id = records.id)", []).map_err(sql)?;
    let missing_preferences: i64 = tx.query_row("SELECT COUNT(*) FROM preferences WHERE id NOT IN (SELECT id FROM reembed_preferences)", [], |r| r.get(0)).map_err(sql)?;
    if missing_preferences > 0 {
        return Err(format!("{missing_preferences} preferences have no new vector; run again"));
    }
    tx.execute("UPDATE preferences SET embedding = (SELECT embedding FROM reembed_preferences WHERE reembed_preferences.id = preferences.id)", []).map_err(sql)?;
    tx.execute("DELETE FROM reembed_preferences", []).map_err(sql)?;
    for (k, v) in [("embedding_model", target.clone()), ("embedding_dims", dims.to_string()), ("previous_embedding_model", current.clone())] {
        tx.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?1, ?2)", params![k, v]).map_err(sql)?;
    }
    tx.execute("DELETE FROM reembed", []).map_err(sql)?;
    tx.execute("DELETE FROM meta WHERE key = ?1", [IN_PROGRESS]).map_err(sql)?;
    tx.commit().map_err(sql)?;
    eprintln!("tidemark reembed: done; {records} records now use {target} ({dims} dimensions). Start the service with the same settings.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inputs_are_rebuilt_as_add_embedded_them() {
        let s = Settings::from_env();
        assert_eq!(embed_input("abc#b0", "user: hello\nassistant: hi", &s), "user: hello\nassistant: hi");
        let long = format!("Caroline: {}", "word ".repeat(700));
        let parts = embed_texts("user", &long, &s);
        assert!(parts.len() > 1);
        assert_eq!(embed_input("abc#0.1", &long, &s), parts[1]);
        assert_eq!(embed_input("abc#0.0", "short one", &s), "short one");
    }
}
