//! Smaller retrieval units (experimental, opt-in: `TIDEMARK_UNITS=on`, per request `units`): records are also indexed as
//! smaller units — one unit per message, and a message longer than `MAX_WORDS` split into windows of whole sentences —
//! each linked to its parent record (and so to the record's timestamp). Search ranks units with the same hybrid method as
//! records (BM25 + cosine, reciprocal-rank fusion, same weights) and scores each record by its best unit; everything after
//! that (user-line ranking, engine-first merge, instruction slots) is unchanged, and the results are the parent records,
//! deduplicated, within the same budget.
//!
//! Units are built by `tidemark units` (resumable; `--estimate` counts without embedding). Records added later have no
//! units until it runs again; they are ranked after the unit-ranked records by the record-level ranking.

use super::*;

/// Words per unit before a message is split into sentence windows.
pub const MAX_WORDS: usize = 80;
/// Target words of a sentence window.
pub const WINDOW_WORDS: usize = 60;

/// The units of a record's content: messages (lines starting with a role or a named speaker), split into sentence windows
/// when long; each unit keeps the message's role or speaker prefix. The date header is not part of any unit.
pub fn split_units(content: &str) -> Vec<String> {
    let mut messages: Vec<(String, String)> = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        let line = match line.strip_prefix('[').and_then(|rest| rest.split_once("] ")) {
            Some((_, rest)) => rest.trim_start(),
            None if line.starts_with('[') && line.ends_with(']') => continue,
            None => line,
        };
        if line.is_empty() {
            continue;
        }
        match line.split_once(": ").filter(|(head, _)| !head.is_empty() && head.len() <= 40 && !head.contains(' ') || head_is_role(head_of(line))) {
            Some((head, body)) => messages.push((format!("{head}: "), body.to_string())),
            None => match messages.last_mut() {
                Some((_, body)) => {
                    body.push(' ');
                    body.push_str(line);
                }
                None => messages.push((String::new(), line.to_string())),
            },
        }
    }
    let mut units = Vec::new();
    for (prefix, body) in messages {
        if body.split_whitespace().count() <= MAX_WORDS {
            units.push(format!("{prefix}{body}"));
            continue;
        }
        let mut window = String::new();
        for sentence in preferences::sentences(&body) {
            if !window.is_empty() && window.split_whitespace().count() + sentence.split_whitespace().count() > WINDOW_WORDS {
                units.push(format!("{prefix}{}", window.trim()));
                window.clear();
            }
            window.push_str(sentence);
            window.push(' ');
        }
        if !window.trim().is_empty() {
            units.push(format!("{prefix}{}", window.trim()));
        }
    }
    units
}

fn head_of(line: &str) -> &str {
    line.split_once(": ").map_or("", |(h, _)| h)
}

fn head_is_role(head: &str) -> bool {
    matches!(head.to_ascii_lowercase().as_str(), "user" | "assistant" | "system" | "tool")
}

pub fn setup(db: &Connection) -> rusqlite::Result<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS units (record_id TEXT NOT NULL, user_id TEXT NOT NULL, idx INTEGER NOT NULL, text TEXT NOT NULL,
            embedding BLOB NOT NULL, PRIMARY KEY (record_id, idx));
         CREATE INDEX IF NOT EXISTS units_user ON units(user_id);",
    )
}

/// A user's units in memory.
pub struct UnitIndex {
    /// (parent record index, tokens, embedding)
    pub units: Vec<(usize, Vec<String>, Vec<f32>)>,
    pub bm25: search::Bm25Index,
}

/// The user's stored units, linked to the loaded records; `None` when the user has none.
pub fn load(db: &Connection, user_id: &str, records: &[Record]) -> rusqlite::Result<Option<UnitIndex>> {
    let position = records.iter().enumerate().map(|(i, r)| (r.id.as_str(), i)).collect::<HashMap<_, _>>();
    let mut stmt = db.prepare("SELECT record_id, text, embedding FROM units WHERE user_id = ?1 ORDER BY record_id, idx")?;
    let rows = stmt.query_map([user_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, Vec<u8>>(2)?)))?;
    let mut index = UnitIndex { units: Vec::new(), bm25: search::Bm25Index::default() };
    for row in rows {
        let (record_id, text, blob) = row?;
        let Some(&i) = position.get(record_id.as_str()) else { continue };
        let tokens = search::tokens(&text);
        index.bm25.add(&tokens);
        index.units.push((i, tokens, blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()));
    }
    Ok((!index.units.is_empty()).then_some(index))
}

/// Records ranked by their best unit (same fusion as records); records without units follow in `fallback` order.
pub fn rank_records(index: &UnitIndex, query_tokens: &[String], qv: &[f32], n_records: usize, s: &Settings, fallback: &[(usize, f32)]) -> Vec<(usize, f32)> {
    let lexical = index.bm25.scores(index.units.iter().map(|(_, t, _)| t.as_slice()), query_tokens);
    let vector = index.units.iter().map(|(_, _, e)| embed::cosine(qv, e)).collect::<Vec<_>>();
    let mut best: Vec<Option<f32>> = vec![None; n_records];
    for (u, score) in search::fuse(&lexical, &vector, s.lexical_weight, s.rrf_k) {
        let r = index.units[u].0;
        if best[r].is_none_or(|b| score > b) {
            best[r] = Some(score);
        }
    }
    let mut ranked = best.iter().enumerate().filter_map(|(i, b)| b.map(|s| (i, s))).collect::<Vec<_>>();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0)));
    let floor = ranked.last().map_or(0.0, |r| r.1);
    let with_units = ranked.iter().map(|(i, _)| *i).collect::<std::collections::HashSet<_>>();
    ranked.extend(fallback.iter().filter(|(i, _)| !with_units.contains(i)).enumerate().map(|(k, (i, _))| (*i, floor - 1e-6 * (k + 1) as f32)));
    ranked
}

/// `tidemark units [--estimate]`: builds the units of every record that has none (embedding them with the configured model).
pub async fn build(args: &[String]) -> Result<(), String> {
    let db_path = std::env::var("TIDEMARK_DB").unwrap_or_else(|_| "tidemark.sqlite".into());
    let mut db = open_db(&db_path).map_err(|e| e.to_string())?;
    let sql = |e: rusqlite::Error| e.to_string();
    let todo = {
        let mut stmt = db
            .prepare("SELECT id, user_id, content FROM records WHERE id NOT IN (SELECT DISTINCT record_id FROM units) ORDER BY seq")
            .map_err(sql)?;
        let rows = stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?, r.get::<_, String>(2)?))).map_err(sql)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(sql)?
    };
    let split = todo.iter().map(|(id, user, content)| (id.clone(), user.clone(), split_units(content))).collect::<Vec<_>>();
    let (n_units, chars) = split.iter().fold((0, 0), |(n, c), (_, _, u)| (n + u.len(), c + u.iter().map(|x| x.chars().count()).sum::<usize>()));
    eprintln!("tidemark units: {} records without units -> {n_units} units, {chars} characters to embed", todo.len());
    if args.iter().any(|a| a == "--estimate") {
        return Ok(());
    }
    let embedder = embed::Embedder::from_env();
    let chunk = std::env::var("TIDEMARK_UNITS_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(50).max(1);
    let mut done = 0;
    for part in split.chunks(chunk) {
        let texts = part.iter().flat_map(|(_, _, u)| u.iter().cloned()).collect::<Vec<_>>();
        let vectors = embedder.embed(&texts, embed::Kind::Document).await?;
        let tx = db.transaction().map_err(sql)?;
        let mut k = 0;
        for (id, user, units) in part {
            for (idx, text) in units.iter().enumerate() {
                let blob = vectors[k].iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                tx.execute("INSERT OR REPLACE INTO units (record_id, user_id, idx, text, embedding) VALUES (?1, ?2, ?3, ?4, ?5)", params![id, user, idx as i64, text, blob])
                    .map_err(sql)?;
                k += 1;
            }
        }
        tx.commit().map_err(sql)?;
        done += part.len();
        eprintln!("tidemark units: {done}/{} records, {} embedding tokens reported", todo.len(), embed::TOKENS_USED.load(std::sync::atomic::Ordering::Relaxed));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_are_units_and_long_ones_are_windowed() {
        let content = "[Monday 8 May 2023, 13:56] \nuser: I love tea.\nassistant: Tea is lovely.";
        assert_eq!(split_units(content), ["user: I love tea.", "assistant: Tea is lovely."]);
        let long = format!("user: {}", (0..30).map(|i| format!("Sentence number {i} is here.")).collect::<Vec<_>>().join(" "));
        let units = split_units(&long);
        assert!(units.len() > 1, "a long message is split");
        assert!(units.iter().all(|u| u.starts_with("user: ")), "each window keeps the role");
        assert!(units.iter().all(|u| u.split_whitespace().count() <= WINDOW_WORDS + 8));
        assert_eq!(units.join(" ").matches("Sentence number").count(), 30, "no sentence lost");
    }
}
