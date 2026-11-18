//! Preference memory: what users like, avoid, habitually do or want, kept as
//! short statements apart from the conversation records, versioned as they
//! change, and each tied to the user's own words.
//!
//! The design follows the preference points of M-flow
//! (https://github.com/FlowElement-xinliuyuansu/m_flow, Apache-2.0: statement,
//! when it applies, evidence, versions); this is an independent implementation.
//!
//! Extraction (`TIDEMARK_PREFERENCES`):
//! - `off` (default): none are extracted (the API still stores preferences given to it).
//! - `rules`: sentences of the user's messages with a preference cue
//!   ("I love …", "I usually …", "my favourite …", "I'm vegetarian", "Always …").
//!   The sentence is the statement and its own evidence. Runs inside Add.
//! - `llm`: the configured language model (`TIDEMARK_LLM_*`) reads the user's
//!   messages after Add returns (a durable queue, resumed after a restart) and
//!   proposes statements, when they apply, an evidence quote, and which earlier
//!   preference a new one replaces. The model only points and names; code checks
//!   that the quote is the user's own words in that record and that a replaced
//!   preference exists, and drops anything else. A batch the model keeps failing
//!   on falls back to the rules.
//!
//! Search returns the most relevant current preferences in
//! `TIDEMARK_PREFERENCE_SLOTS` final slots (default 0 = none; per request
//! `preference_slots`), marked `"kind": "preference"`.

use super::*;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Rules,
    Llm,
}

#[derive(Clone)]
pub struct Config {
    pub mode: Mode,
    pub slots: usize,
    /// Characters of user messages per model call.
    pub batch_chars: usize,
    /// Users processed by the model at the same time.
    pub workers: usize,
    /// Existing preferences shown to the model (the most related ones).
    pub context: usize,
    /// Model attempts on a batch before it falls back to the rules.
    pub attempts: i64,
    /// Seconds new records wait, so one model call covers several Adds.
    pub delay: i64,
    pub prompt: String,
    /// How preferences are labelled in search results: `plain` (all "[preference]"),
    /// `kind` (also "[the user already does this]"), `novelty` (kind, and for a request
    /// for something new also "not new to them"). Per request: `preference_labels`.
    pub labels: Labels,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Labels {
    Plain,
    Kind,
    Novelty,
}

impl Labels {
    pub fn parse(v: &str) -> Option<Labels> {
        match v.trim().to_ascii_lowercase().as_str() {
            "plain" => Some(Labels::Plain),
            "kind" => Some(Labels::Kind),
            "novelty" => Some(Labels::Novelty),
            _ => None,
        }
    }
}

impl Config {
    pub fn from_env() -> Result<Self, String> {
        let var = |k: &str| std::env::var(k).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
        let num = |k: &str, d: usize| var(k).and_then(|v| v.parse().ok()).unwrap_or(d);
        let mode = match var("TIDEMARK_PREFERENCES").map(|v| v.to_ascii_lowercase()).as_deref() {
            Some("rules") => Mode::Rules,
            Some("llm") | Some("model") => Mode::Llm,
            None | Some("off") | Some("0") | Some("none") => Mode::Off,
            Some(other) => return Err(format!("TIDEMARK_PREFERENCES must be rules, llm or off, not {other:?}")),
        };
        let prompt = match var("TIDEMARK_PREFERENCE_PROMPT_FILE") {
            Some(path) => std::fs::read_to_string(&path).map_err(|e| format!("TIDEMARK_PREFERENCE_PROMPT_FILE {path}: {e}"))?,
            None => DEFAULT_PROMPT.to_string(),
        };
        Ok(Self {
            mode,
            slots: num("TIDEMARK_PREFERENCE_SLOTS", 0),
            batch_chars: num("TIDEMARK_PREFERENCE_BATCH_CHARS", 12000).max(1),
            workers: num("TIDEMARK_PREFERENCE_WORKERS", 2).max(1),
            context: num("TIDEMARK_PREFERENCE_CONTEXT", 50),
            attempts: num("TIDEMARK_PREFERENCE_ATTEMPTS", 4).max(1) as i64,
            delay: num("TIDEMARK_PREFERENCE_DELAY", 10) as i64,
            labels: match var("TIDEMARK_PREFERENCE_LABELS") {
                Some(v) => Labels::parse(&v).ok_or_else(|| format!("TIDEMARK_PREFERENCE_LABELS must be plain, kind or novelty, not {v:?}"))?,
                None => Labels::Plain,
            },
            prompt,
        })
    }
}

const DEFAULT_PROMPT: &str = "You read messages a user wrote to an assistant and note the user's lasting preferences: \
what they like or dislike, habits and routines, constraints (diet, allergies, health, schedule, budget, accessibility), \
how they want answers (style, format, language, tone), and standing instructions. Do not note one-off requests, facts \
about other people, or anything only the assistant said.\n\
For each preference give:\n\
- record: the label of the message it comes from (R1, R2, ...);\n\
- statement: a short statement in the third person, starting \"The user\";\n\
- when_to_apply: the situations in which it matters, in a few words;\n\
- category: one or two words, e.g. food, activity, schedule, communication, work, health, travel, shopping, entertainment;\n\
- kind: \"likes\" (a taste: something the user likes, dislikes or avoids), \"does\" (something the user already does or \
has done, e.g. a hobby, routine or tradition), or \"instruction\" (how the assistant should act). If both a taste and a \
practice are stated, give two entries;\n\
- evidence: the user's own words that show it, copied exactly, the shortest span that suffices;\n\
- replaces: the id of an existing preference that this one updates or contradicts, or null.\n\
Existing preferences are listed with their ids; do not repeat one that still holds. If a message changes or reverses \
an existing preference, give the new one with replaces set to the old id.\n\
Reply with one JSON object: {\"preferences\": [ ... ]}. Use an empty list when there are none.";

pub fn setup(db: &Connection) -> rusqlite::Result<()> {
    db.execute_batch(
        "CREATE TABLE IF NOT EXISTS preferences (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, seq INTEGER NOT NULL,
            statement TEXT NOT NULL, when_to_apply TEXT, category TEXT, evidence TEXT, record_id TEXT, source TEXT NOT NULL,
            status TEXT NOT NULL, version INTEGER NOT NULL, replaces TEXT, stated_at TEXT, updated_at TEXT NOT NULL, embedding BLOB NOT NULL);
         CREATE INDEX IF NOT EXISTS preferences_user ON preferences(user_id, status);
         CREATE INDEX IF NOT EXISTS preferences_record ON preferences(user_id, record_id);
         CREATE TABLE IF NOT EXISTS preference_queue (record_id TEXT PRIMARY KEY, user_id TEXT NOT NULL, seq INTEGER NOT NULL,
            text TEXT NOT NULL, stated_at TEXT, attempts INTEGER NOT NULL DEFAULT 0, next_try INTEGER NOT NULL DEFAULT 0);
         CREATE INDEX IF NOT EXISTS preference_queue_user ON preference_queue(user_id, seq);",
    )?;
    // Added after the table's first version: a database from before gets the column.
    let has_kind = db.prepare("SELECT kind FROM preferences LIMIT 0").is_ok();
    if !has_kind {
        db.execute_batch("ALTER TABLE preferences ADD COLUMN kind TEXT;")?;
    }
    Ok(())
}

/// A proposed preference, before it is embedded and stored.
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub record_id: Option<String>,
    pub statement: String,
    pub when_to_apply: Option<String>,
    pub category: Option<String>,
    pub evidence: Option<String>,
    pub stated_at: Option<String>,
    pub source: &'static str,
    pub replaces: Option<String>,
    /// `likes` (a taste), `does` (something the user already does) or `instruction`.
    pub kind: Option<String>,
}

/// The model's kind in one of the three names (unknown words are dropped).
pub fn normalize_kind(kind: &str) -> Option<String> {
    let k = kind.trim().to_ascii_lowercase();
    let k = match k.as_str() {
        "likes" | "like" | "taste" | "preference" | "dislikes" | "dislike" | "avoids" | "avoid" => "likes",
        "does" | "do" | "practice" | "habit" | "routine" | "activity" | "tradition" | "already does" => "does",
        "instruction" | "instructions" | "rule" => "instruction",
        _ => return None,
    };
    Some(k.to_string())
}

impl Candidate {
    /// The text that is embedded and ranked: statement and situations.
    pub fn index_text(&self) -> String {
        match &self.when_to_apply {
            Some(w) => format!("{} (applies when: {w})", self.statement),
            None => self.statement.clone(),
        }
    }
}

/// Lower case, one space, plain quotes: for comparing quotes and statements.
pub fn normalize(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '’' | '‘' | '`' => '\'',
            '“' | '”' => '"',
            c => c.to_ascii_lowercase(),
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim_matches(|c: char| c.is_ascii_punctuation() && c != '\'' && c != '"')
        .to_string()
}

/// Whether `evidence` is the user's own words in `text` (case, spacing,
/// quote styles and surrounding punctuation ignored).
pub fn evidence_holds(evidence: &str, text: &str) -> bool {
    let e = normalize(evidence);
    e.chars().filter(|c| c.is_alphanumeric()).count() >= 3 && normalize(text).contains(&e)
}

/// The user's text in each new record: the user messages a record holds
/// (block records join several; a long message's parts are records of their own).
pub fn user_texts(rows: &[(String, String, Option<String>)], user_messages: &[String]) -> Vec<(String, String, Option<String>)> {
    rows.iter()
        .filter_map(|(id, raw, created_at)| {
            let mut parts = user_messages.iter().map(|m| m.trim()).filter(|m| !m.is_empty() && raw.contains(*m)).collect::<Vec<_>>();
            if parts.is_empty() {
                // A part of a long message (possibly led by the speaker).
                let body = raw.split_once(": ").filter(|(head, _)| head.len() <= 40 && !head.contains('\n')).map_or(raw.as_str(), |(_, b)| b);
                if user_messages.iter().any(|m| m.contains(body.trim())) && !body.trim().is_empty() {
                    parts.push(raw.trim());
                }
            }
            (!parts.is_empty()).then(|| (id.clone(), parts.join("\n"), created_at.clone()))
        })
        .collect()
}

/// Sentences of a text (split at . ! ? followed by a space, and at line breaks).
pub fn sentences(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut start = 0;
        let bytes = line.as_bytes();
        for (i, c) in line.char_indices() {
            if matches!(c, '.' | '!' | '?') && bytes.get(i + 1).is_none_or(|b| b.is_ascii_whitespace()) {
                out.push(line[start..=i].trim());
                start = i + 1;
            }
        }
        out.push(line[start..].trim());
    }
    out.into_iter().filter(|s| !s.is_empty()).collect()
}

/// Whether a sentence states a preference, habit, constraint or standing instruction.
pub fn is_preference_sentence(sentence: &str) -> bool {
    let words = sentence.split_whitespace().count();
    (3..=80).contains(&words) && !sentence.trim_end().ends_with('?') && !is_reaction(sentence) && (taste_cue(sentence) || is_habit(sentence) || search::is_standing_instruction(sentence))
}

/// "I love …", "my favourite …", "I'm vegetarian": a stated taste or trait.
pub fn taste_cue(sentence: &str) -> bool {
    static CUE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let cue = CUE.get_or_init(|| {
        regex::Regex::new(concat!(
            r"(?i)\bi(?:'m|’m| am)?(?: really| truly| absolutely| just| also| generally| usually| still| do)? ",
            r"(?:love|like|enjoy|prefer|adore|hate|dislike|detest|can't stand|can’t stand|cannot stand|don't like|don’t like|do not like|",
            r"don't eat|don’t eat|never eat|avoid|allergic to|intolerant to)\b",
            r"|\bmy (?:favou?rite|go-to|preferred)\b",
            r"|\bi(?:'m|’m| am) (?:a |an )?(?:vegetarian|vegan|pescatarian|lactose[- ]intolerant|gluten[- ]free|allergic|night owl|morning person|early riser|introvert|extrovert)\b",
        ))
        .unwrap()
    });
    cue.is_match(sentence)
}

/// Whether a question asks for something new ("new ideas", "something different",
/// "haven't tried"): then what the user already does is marked as not new.
pub fn asks_for_novelty(query: &str) -> bool {
    static NEW: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    NEW.get_or_init(|| {
        regex::Regex::new(concat!(
            r"(?i)\b(?:new|different|fresh|other|alternative)\s+(?:ideas?|ways?|things?|activit\w+|hobb\w+|options?|suggestions?|recommendations?|experiences?|approach\w*|places?)\b",
            r"|\bsomething (?:new|different|else)\b|\bhaven'?t tried\b|\bnever tried\b|\btry (?:out )?something\b|\bbranch out\b|\bswitch (?:it|things) up\b|\bchange things up\b",
        ))
        .unwrap()
    })
    .is_match(query)
}

/// "I usually start early", "I always bring snacks": a habit. Narration with the
/// same words is not ("I often find myself wondering …", "I never expected …",
/// "I usually thought …"): the word after the adverb must not be a verb of
/// noticing, thinking or happening, nor in the past tense.
pub fn is_habit(sentence: &str) -> bool {
    static HABIT: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    const NARRATION: &[&str] = &[
        "find", "found", "expected", "thought", "realized", "realised", "wonder", "wondered", "considered", "felt", "feel", "left", "reflect",
        "reflected", "imagined", "dreamed", "daydream", "worry", "worried", "struggle", "struggled", "get", "got", "catch", "caught", "end",
        "ended", "had", "was", "knew", "believed", "noticed", "came", "saw", "heard",
    ];
    let habit = HABIT.get_or_init(|| regex::Regex::new(r"(?i)\bi (?:usually|always|never|tend to|typically|normally|rarely|often) (\w+)").unwrap());
    habit.captures_iter(sentence).any(|c| {
        let next = c[1].to_ascii_lowercase();
        !NARRATION.contains(&next.as_str()) && !next.ends_with("ed")
    })
}

/// A short reaction to what was just said ("I love that!", "Oh, I really like this idea."):
/// its object is only "that", "this" or "it", so it states nothing lasting.
pub fn is_reaction(sentence: &str) -> bool {
    static REACTION: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    REACTION
        .get_or_init(|| {
            regex::Regex::new(concat!(
                r"(?i)^\W*(?:(?:oh|wow|yes|yeah|haha|aw|ah|ok|okay|great|thanks|thank you)\W+)*",
                r"i(?:'d|’d| would)?(?: really| just| absolutely| totally| truly| do)? (?:love|like|enjoy|adore|hate|dislike|prefer) ",
                r"(?:that|this|it|those|these)(?: (?:idea|suggestion|one|plan|approach|part|so much|a lot))?\W*$",
            ))
            .unwrap()
        })
        .is_match(sentence.trim())
}

/// Preferences found by the rules: each matching sentence, verbatim.
pub fn rule_candidates(record_id: &str, text: &str, stated_at: Option<&str>) -> Vec<Candidate> {
    let mut seen = HashSet::new();
    sentences(text)
        .into_iter()
        .filter(|s| is_preference_sentence(s))
        .filter(|s| seen.insert(normalize(s)))
        .map(|s| Candidate {
            record_id: Some(record_id.to_string()),
            statement: s.to_string(),
            when_to_apply: None,
            category: Some(if search::is_standing_instruction(s) { "instruction" } else { "preference" }.to_string()),
            evidence: Some(s.to_string()),
            stated_at: stated_at.map(String::from),
            source: "rules",
            replaces: None,
            kind: Some(
                if search::is_standing_instruction(s) {
                    "instruction"
                } else if is_habit(s) && !taste_cue(s) {
                    "does"
                } else {
                    "likes"
                }
                .to_string(),
            ),
        })
        .collect()
}

fn now() -> String {
    Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Stores embedded candidates: a statement equal to a current one is skipped,
/// a replacement supersedes the current preference it names (a new version).
/// Returns how many were stored.
pub fn store(tx: &rusqlite::Transaction, user_id: &str, items: &[(Candidate, Vec<f32>)]) -> rusqlite::Result<usize> {
    // Current statements by id (normalized), to skip repeats.
    let mut current = {
        let mut stmt = tx.prepare("SELECT id, statement FROM preferences WHERE user_id = ?1 AND status = 'current'")?;
        let rows = stmt.query_map([user_id], |r| Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?.into_iter().map(|(id, s)| (id, normalize(&s))).collect::<HashMap<_, _>>()
    };
    let mut stored = 0;
    for (c, vector) in items {
        let statement = normalize(&c.statement);
        // A repeat of a current statement is skipped, unless it is a new
        // version of that very preference (e.g. only its situations changed).
        if current.iter().any(|(id, s)| *s == statement && c.replaces.as_ref() != Some(id)) {
            continue;
        }
        if let Some(rid) = &c.record_id {
            let exists = tx.query_row("SELECT 1 FROM records WHERE id = ?1 AND user_id = ?2", params![rid, user_id], |_| Ok(())).is_ok();
            if !exists {
                continue; // the record was deleted meanwhile
            }
        }
        let replaced: Option<(String, i64)> = match &c.replaces {
            Some(old) => tx
                .query_row("SELECT id, version FROM preferences WHERE id = ?1 AND user_id = ?2 AND status = 'current'", params![old, user_id], |r| Ok((r.get(0)?, r.get(1)?)))
                .ok(),
            None => None,
        };
        let seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq), 0) + 1 FROM preferences", [], |r| r.get(0))?;
        let id = format!("pref_{}", short_hash(&format!("{user_id}\u{0}{}\u{0}{}\u{0}{seq}", c.record_id.as_deref().unwrap_or(""), c.statement)));
        if let Some((old, _)) = &replaced {
            tx.execute("UPDATE preferences SET status = 'superseded', updated_at = ?1 WHERE id = ?2", params![now(), old])?;
        }
        if let Some((old, _)) = &replaced {
            current.remove(old);
        }
        current.insert(id.clone(), statement);
        let blob = vector.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        tx.execute(
            "INSERT INTO preferences (id, user_id, seq, statement, when_to_apply, category, evidence, record_id, source, status, version, replaces, stated_at, updated_at, embedding, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'current', ?10, ?11, ?12, ?13, ?14, ?15)",
            params![
                id,
                user_id,
                seq,
                c.statement,
                c.when_to_apply,
                c.category,
                c.evidence,
                c.record_id,
                c.source,
                replaced.as_ref().map_or(1, |(_, v)| v + 1),
                replaced.as_ref().map(|(old, _)| old.clone()),
                c.stated_at,
                now(),
                blob,
                c.kind
            ],
        )?;
        stored += 1;
    }
    Ok(stored)
}

/// Queues records for the model, due after `delay` seconds.
pub fn enqueue(tx: &rusqlite::Transaction, user_id: &str, texts: &[(String, String, Option<String>)], delay: i64) -> rusqlite::Result<()> {
    let seq: i64 = tx.query_row("SELECT COALESCE(MAX(seq), 0) + 1 FROM preference_queue", [], |r| r.get(0))?;
    for (i, (record_id, text, stated_at)) in texts.iter().enumerate() {
        tx.execute(
            "INSERT OR REPLACE INTO preference_queue (record_id, user_id, seq, text, stated_at, attempts, next_try) VALUES (?1, ?2, ?3, ?4, ?5, 0, ?6)",
            params![record_id, user_id, seq + i as i64, text, stated_at, Utc::now().timestamp() + delay],
        )?;
    }
    Ok(())
}

/// A record was deleted or corrected: preferences drawn from it no longer
/// stand on it, and its pending extraction is dropped.
pub fn retract_record(tx: &rusqlite::Transaction, user_id: &str, record_id: &str) -> rusqlite::Result<usize> {
    tx.execute("DELETE FROM preference_queue WHERE user_id = ?1 AND record_id = ?2", params![user_id, record_id])?;
    tx.execute(
        "UPDATE preferences SET status = 'retracted', updated_at = ?1 WHERE user_id = ?2 AND record_id = ?3 AND status = 'current'",
        params![now(), user_id, record_id],
    )
}

pub fn delete_user(tx: &rusqlite::Transaction, user_id: &str) -> rusqlite::Result<()> {
    tx.execute("DELETE FROM preferences WHERE user_id = ?1", [user_id])?;
    tx.execute("DELETE FROM preference_queue WHERE user_id = ?1", [user_id])?;
    Ok(())
}

/// A stored preference.
#[derive(Clone, Debug)]
pub struct Preference {
    pub id: String,
    pub statement: String,
    pub when_to_apply: Option<String>,
    pub category: Option<String>,
    pub evidence: Option<String>,
    pub record_id: Option<String>,
    pub source: String,
    pub status: String,
    pub version: i64,
    pub replaces: Option<String>,
    pub stated_at: Option<String>,
    pub updated_at: String,
    pub embedding: Vec<f32>,
    pub kind: Option<String>,
}

const COLUMNS: &str = "id, statement, when_to_apply, category, evidence, record_id, source, status, version, replaces, stated_at, updated_at, embedding, kind";

fn row(r: &rusqlite::Row) -> rusqlite::Result<Preference> {
    let blob: Vec<u8> = r.get(12)?;
    Ok(Preference {
        id: r.get(0)?,
        statement: r.get(1)?,
        when_to_apply: r.get(2)?,
        category: r.get(3)?,
        evidence: r.get(4)?,
        record_id: r.get(5)?,
        source: r.get(6)?,
        status: r.get(7)?,
        version: r.get(8)?,
        replaces: r.get(9)?,
        stated_at: r.get(10)?,
        updated_at: r.get(11)?,
        embedding: blob.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        kind: r.get(13)?,
    })
}

/// A user's preferences in the order they were stored; `all` includes
/// superseded, retracted and deleted ones.
pub fn load(db: &Connection, user_id: &str, all: bool) -> rusqlite::Result<Vec<Preference>> {
    let sql = format!("SELECT {COLUMNS} FROM preferences WHERE user_id = ?1 {} ORDER BY seq", if all { "" } else { "AND status = 'current'" });
    let mut stmt = db.prepare(&sql)?;
    let rows = stmt.query_map([user_id], row)?;
    rows.collect()
}

pub fn get(db: &Connection, user_id: &str, id: &str) -> rusqlite::Result<Preference> {
    db.query_row(&format!("SELECT {COLUMNS} FROM preferences WHERE user_id = ?1 AND id = ?2"), params![user_id, id], row)
}

impl Preference {
    pub fn json(&self) -> serde_json::Value {
        serde_json::json!({
            "id": self.id, "statement": self.statement, "when_to_apply": self.when_to_apply, "category": self.category,
            "evidence": self.evidence, "record_id": self.record_id, "source": self.source, "status": self.status,
            "version": self.version, "replaces": self.replaces, "stated_at": self.stated_at, "updated_at": self.updated_at, "kind": self.kind,
        })
    }

    fn index_text(&self) -> String {
        match &self.when_to_apply {
            Some(w) => format!("{} (applies when: {w})", self.statement),
            None => self.statement.clone(),
        }
    }

    /// The search result text: the statement (the model's wording when a model
    /// extracted it), when it applies, and the user's own words as evidence.
    /// Something the user already does is labelled as such, and for a request
    /// for something new (`novelty`) also as not new to them.
    pub fn render(&self, labels: Labels, novelty: bool) -> String {
        let date = self.stated_at.as_deref().and_then(|d| DateTime::parse_from_rfc3339(d).ok()).map(|d| format!(", stated {}", d.format("%-d %B %Y")));
        let label = match (labels, self.kind.as_deref()) {
            (Labels::Plain, _) => "preference",
            (Labels::Novelty, Some("does")) if novelty => "the user already does this; not new to them",
            (_, Some("does")) => "the user already does this",
            (_, Some("instruction")) => "instruction",
            _ => "preference",
        };
        let mut text = format!("[{label}{}] {}", date.unwrap_or_default(), self.statement);
        if let Some(note) = preference_note() {
            text.push_str(&format!("\n{note}"));
        }
        if let Some(w) = &self.when_to_apply {
            text.push_str(&format!("\napplies when: {w}"));
        }
        if let Some(e) = self.evidence.as_ref().filter(|e| normalize(e) != normalize(&self.statement)) {
            text.push_str(&format!("\nuser's words: \"{e}\""));
        }
        text
    }
}

/// `TIDEMARK_PREFERENCE_NOTE`: an optional line added to every preference in
/// search results (e.g. that it is background, not a request).
fn preference_note() -> Option<&'static str> {
    static NOTE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    NOTE.get_or_init(|| std::env::var("TIDEMARK_PREFERENCE_NOTE").ok().map(|n| n.trim().to_string()).filter(|n| !n.is_empty())).as_deref()
}

/// The `n` current preferences most relevant to the query (BM25 over statement
/// and situations, fused with cosine similarity, as for records).
pub fn rank(prefs: &[Preference], query_tokens: &[String], qv: &[f32], n: usize, s: &Settings) -> Vec<(usize, f32)> {
    if n == 0 || prefs.is_empty() {
        return Vec::new();
    }
    let toks = prefs.iter().map(|p| search::tokens(&p.index_text())).collect::<Vec<_>>();
    let bm = search::Bm25::new(toks.iter().map(|t| t.as_slice()).collect());
    let lex = bm.scores(query_tokens);
    let vec = prefs.iter().map(|p| embed::cosine(qv, &p.embedding)).collect::<Vec<_>>();
    search::fuse(&lex, &vec, s.lexical_weight, s.rrf_k).into_iter().take(n).collect()
}

/// Rules extraction for new records, embedded (used by Add and as the model's fallback).
pub async fn rules_for(embedder: &embed::Embedder, texts: &[(String, String, Option<String>)]) -> Result<Vec<(Candidate, Vec<f32>)>, String> {
    let candidates = texts.iter().flat_map(|(id, text, at)| rule_candidates(id, text, at.as_deref())).collect::<Vec<_>>();
    embedded(embedder, candidates).await
}

pub async fn embedded(embedder: &embed::Embedder, candidates: Vec<Candidate>) -> Result<Vec<(Candidate, Vec<f32>)>, String> {
    if candidates.is_empty() {
        return Ok(Vec::new());
    }
    let vectors = embedder.embed(&candidates.iter().map(Candidate::index_text).collect::<Vec<_>>(), embed::Kind::Document).await?;
    Ok(candidates.into_iter().zip(vectors).collect())
}

// ---- Extraction by the language model -------------------------------------------------

/// One queued record.
#[derive(Clone, Debug)]
pub struct Pending {
    pub record_id: String,
    pub text: String,
    pub stated_at: Option<String>,
    pub attempts: i64,
}

/// The model's reply checked against the batch: proposals whose record label is
/// unknown, whose evidence is not the user's words in that record, or without
/// a statement are dropped; `replaces` is kept only when it names a current
/// preference. Returns the kept candidates and the number dropped.
#[cfg(test)]
pub fn check_reply(reply: &serde_json::Value, batch: &[Pending], current_ids: &HashSet<String>) -> (Vec<Candidate>, usize) {
    let (kept, verdicts) = check_reply_audited(reply, batch, current_ids);
    let dropped = verdicts.iter().filter(|v| v["verdict"] != "kept").count();
    (kept, dropped)
}

/// `check_reply` with one verdict per proposal (for `TIDEMARK_PREFERENCE_AUDIT_FILE`): `kept`, or the specific
/// reason it was rejected (`unknown_record`, `missing_statement`, `missing_evidence`, `quote_not_verbatim`), with the
/// proposal as the model gave it and the record's user text.
pub fn check_reply_audited(reply: &serde_json::Value, batch: &[Pending], current_ids: &HashSet<String>) -> (Vec<Candidate>, Vec<serde_json::Value>) {
    let items = reply.get("preferences").and_then(|v| v.as_array()).or_else(|| reply.as_array()).cloned().unwrap_or_default();
    let mut verdicts = Vec::new();
    let text_of = |v: &serde_json::Value, k: &str| v.get(k).and_then(|x| x.as_str()).map(str::trim).filter(|x| !x.is_empty() && !x.eq_ignore_ascii_case("null")).map(String::from);
    let mut kept = Vec::new();
    for item in items {
        let label = item.get("record").map(|r| r.as_str().map(String::from).unwrap_or_else(|| r.to_string())).unwrap_or_default();
        let index = label.trim().trim_start_matches(['R', 'r']).parse::<usize>().ok().and_then(|i| i.checked_sub(1));
        let record = index.and_then(|i| batch.get(i));
        let mut verdict = |v: &str| {
            verdicts.push(serde_json::json!({
                "verdict": v, "proposal": item, "record_id": record.map(|p| p.record_id.clone()),
                "record_text": record.map(|p| p.text.clone()), "stated_at": record.and_then(|p| p.stated_at.clone()),
            }))
        };
        let (Some(p), Some(statement), Some(evidence)) = (record, text_of(&item, "statement"), text_of(&item, "evidence")) else {
            verdict(if record.is_none() { "unknown_record" } else if text_of(&item, "statement").is_none() { "missing_statement" } else { "missing_evidence" });
            continue;
        };
        if !evidence_holds(&evidence, &p.text) {
            verdict("quote_not_verbatim");
            continue;
        }
        verdict("kept");
        kept.push(Candidate {
            record_id: Some(p.record_id.clone()),
            statement,
            when_to_apply: text_of(&item, "when_to_apply"),
            category: text_of(&item, "category"),
            evidence: Some(evidence),
            stated_at: p.stated_at.clone(),
            source: "llm",
            replaces: text_of(&item, "replaces").filter(|id| current_ids.contains(id)),
            kind: text_of(&item, "kind").and_then(|k| normalize_kind(&k)),
        });
    }
    (kept, verdicts)
}

/// Appends audit verdicts (JSON lines) when `TIDEMARK_PREFERENCE_AUDIT_FILE` is set.
fn write_audit(user_id: &str, verdicts: &[serde_json::Value]) {
    static FILE: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(path) = FILE.get_or_init(|| std::env::var("TIDEMARK_PREFERENCE_AUDIT_FILE").ok().filter(|p| !p.trim().is_empty())) else { return };
    use std::io::Write;
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
        for v in verdicts {
            let mut v = v.clone();
            v["user_id"] = serde_json::json!(user_id);
            let _ = writeln!(f, "{v}");
        }
    }
}

/// The next user with pending records (not being processed, due now) and a
/// batch of their records up to `batch_chars`.
fn claim(state: &AppState) -> Option<(String, Vec<Pending>)> {
    let now = Utc::now().timestamp();
    let db = state.db.lock().unwrap();
    let users = {
        let mut stmt = db.prepare("SELECT user_id FROM preference_queue WHERE next_try <= ?1 GROUP BY user_id ORDER BY MIN(seq) LIMIT 64").ok()?;
        let rows = stmt.query_map([now], |r| r.get::<_, String>(0)).ok()?;
        rows.filter_map(Result::ok).collect::<Vec<_>>()
    };
    let mut busy = state.preference_busy.lock().unwrap();
    let user = users.into_iter().find(|u| !busy.contains(u))?;
    // Once the oldest record is due, newer ones join it; records waiting for a retry do not.
    let mut stmt = db.prepare("SELECT record_id, text, stated_at, attempts FROM preference_queue WHERE user_id = ?1 AND (attempts = 0 OR next_try <= ?2) ORDER BY seq").ok()?;
    let rows = stmt
        .query_map(params![user, now], |r| Ok(Pending { record_id: r.get(0)?, text: r.get(1)?, stated_at: r.get(2)?, attempts: r.get(3)? }))
        .ok()?
        .filter_map(Result::ok);
    let mut batch = Vec::new();
    let mut chars = 0;
    for p in rows {
        chars += p.text.chars().count();
        if !batch.is_empty() && chars > state.preferences.batch_chars {
            break;
        }
        batch.push(p);
    }
    busy.insert(user.clone());
    Some((user, batch))
}

/// Runs `workers` extraction loops; `preference_wake` is notified by Add.
pub fn spawn(state: Arc<AppState>) {
    for _ in 0..state.preferences.workers {
        let state = state.clone();
        tokio::spawn(async move {
            loop {
                match claim(&state) {
                    Some((user, batch)) => {
                        process(&state, &user, &batch).await;
                        state.preference_busy.lock().unwrap().remove(&user);
                    }
                    None => {
                        // Until the next queued record is due (at most 30 s), or until Add wakes us.
                        let due: Option<i64> = state.db.lock().unwrap().query_row("SELECT MIN(next_try) FROM preference_queue", [], |r| r.get(0)).ok().flatten();
                        let wait = due.map_or(30, |d| (d - Utc::now().timestamp()).clamp(1, 30)) as u64;
                        let _ = tokio::time::timeout(std::time::Duration::from_secs(wait), state.preference_wake.notified()).await;
                    }
                }
            }
        });
    }
}

async fn process(state: &AppState, user_id: &str, batch: &[Pending]) {
    let Some(llm) = state.llm.as_ref() else { return };
    let current = match load(&state.db.lock().unwrap(), user_id, false) {
        Ok(c) => c,
        Err(e) => return eprintln!("preferences user={user_id}: {e}"),
    };
    // The existing preferences most related to these messages.
    let batch_text = batch.iter().map(|p| p.text.as_str()).collect::<Vec<_>>().join("\n");
    let shown = if current.len() <= state.preferences.context {
        (0..current.len()).collect::<Vec<_>>()
    } else {
        let toks = current.iter().map(|p| search::tokens(&p.index_text())).collect::<Vec<_>>();
        let bm = search::Bm25::new(toks.iter().map(|t| t.as_slice()).collect());
        let mut scored = bm.scores(&search::tokens(&batch_text)).into_iter().enumerate().collect::<Vec<_>>();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.into_iter().take(state.preferences.context).map(|(i, _)| i).collect()
    };
    let existing = shown.iter().map(|&i| serde_json::json!({"id": current[i].id, "statement": current[i].statement})).collect::<Vec<_>>();
    let messages = batch
        .iter()
        .enumerate()
        .map(|(i, p)| serde_json::json!({"record": format!("R{}", i + 1), "date": p.stated_at, "text": p.text}))
        .collect::<Vec<_>>();
    let input = serde_json::json!({"existing_preferences": existing, "messages": messages}).to_string();
    let current_ids = current.iter().map(|p| p.id.clone()).collect::<HashSet<_>>();
    let before = llm.usage();
    let outcome = match llm.complete_json(&state.preferences.prompt, &input, None).await {
        Ok(reply) => {
            let (candidates, verdicts) = check_reply_audited(&reply, batch, &current_ids);
            write_audit(user_id, &verdicts);
            let dropped = verdicts.iter().filter(|v| v["verdict"] != "kept").count();
            embedded(&state.embedder, candidates).await.map(|c| (c, dropped))
        }
        Err(e) => Err(e),
    };
    let after = llm.usage();
    let ids = batch.iter().map(|p| p.record_id.clone()).collect::<Vec<_>>();
    if matches!(&outcome, Err(e) if e == "llm_attempt_cap_reached") {
        // The operator's cap: stop extracting, keep the queue, no fallback to the rules.
        eprintln!("preferences: language-model attempt cap reached; extraction stopped, {} records stay queued llm_attempts_total={}", batch.len(), after.attempts);
        tokio::time::sleep(std::time::Duration::from_secs(300)).await;
        return;
    }
    match outcome {
        Ok((items, dropped)) => {
            let stored = finish(state, user_id, &ids, &items);
            eprintln!(
                "preferences user={user_id} records={} proposed={} dropped={dropped} stored={stored} llm_tokens_in={} llm_tokens_out={} llm_attempts_total={}",
                batch.len(),
                items.len() + dropped,
                after.input_tokens - before.input_tokens,
                after.output_tokens - before.output_tokens,
                after.attempts
            );
        }
        Err(e) if batch.iter().map(|p| p.attempts).max().unwrap_or(0) + 1 >= state.preferences.attempts => {
            // The fallback ladder: the rules, so these records are not skipped.
            let texts = batch.iter().map(|p| (p.record_id.clone(), p.text.clone(), p.stated_at.clone())).collect::<Vec<_>>();
            match rules_for(&state.embedder, &texts).await {
                Ok(items) => {
                    let stored = finish(state, user_id, &ids, &items);
                    eprintln!("preferences user={user_id} records={}: model failed ({e}); rules stored {stored} llm_attempts_total={}", batch.len(), after.attempts);
                }
                Err(e2) => {
                    eprintln!("preferences user={user_id}: model failed ({e}) and rules could not be embedded ({e2}); retried later llm_attempts_total={}", after.attempts);
                    postpone(state, user_id, batch);
                }
            }
        }
        Err(e) => {
            eprintln!("preferences user={user_id} records={}: {e}; retried later llm_attempts_total={}", batch.len(), after.attempts);
            postpone(state, user_id, batch);
        }
    }
}

/// Stores the batch's preferences and removes it from the queue.
fn finish(state: &AppState, user_id: &str, ids: &[String], items: &[(Candidate, Vec<f32>)]) -> usize {
    let mut db = state.db.lock().unwrap();
    let result = (|| -> rusqlite::Result<usize> {
        let tx = db.transaction()?;
        // Only records still queued: one deleted or corrected meanwhile was dequeued.
        let queued = ids
            .iter()
            .filter(|id| tx.query_row("SELECT 1 FROM preference_queue WHERE record_id = ?1 AND user_id = ?2", params![id, user_id], |_| Ok(())).is_ok())
            .cloned()
            .collect::<HashSet<_>>();
        let items = items.iter().filter(|(c, _)| c.record_id.as_ref().is_some_and(|r| queued.contains(r))).cloned().collect::<Vec<_>>();
        let n = store(&tx, user_id, &items)?;
        for id in &queued {
            tx.execute("DELETE FROM preference_queue WHERE record_id = ?1 AND user_id = ?2", params![id, user_id])?;
        }
        tx.commit()?;
        Ok(n)
    })();
    result.unwrap_or_else(|e| {
        eprintln!("preferences user={user_id}: storing failed: {e}");
        0
    })
}

fn postpone(state: &AppState, user_id: &str, batch: &[Pending]) {
    let db = state.db.lock().unwrap();
    for p in batch {
        let delay = 30i64 << p.attempts.min(5);
        let _ = db.execute(
            "UPDATE preference_queue SET attempts = attempts + 1, next_try = ?1 WHERE record_id = ?2 AND user_id = ?3",
            params![Utc::now().timestamp() + delay, p.record_id, user_id],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn rules_find_preference_sentences_verbatim() {
        let text = "Hi there! I really love hiking in the mountains. Can you book a table?\nI'm vegetarian, so no meat please.\n\
                    The weather was nice. My favourite tea is sencha. From now on, answer in French. Do I like jazz?";
        let found = rule_candidates("r1", text, Some("2023-05-08T13:56:00Z")).into_iter().map(|c| c.statement).collect::<Vec<_>>();
        assert_eq!(found, ["I really love hiking in the mountains.", "I'm vegetarian, so no meat please.", "My favourite tea is sencha.", "From now on, answer in French."]);
        assert!(!is_preference_sentence("I'd like to book a table for two."), "a request is not a preference");
        for reaction in ["I love that!", "Oh, I really like this idea.", "Yes, I love it so much!", "I like that!"] {
            assert!(!is_preference_sentence(reaction), "{reaction}");
        }
        assert!(is_preference_sentence("I love that little bakery near the station."), "a real object stays");
        assert!(is_preference_sentence("I usually run before work."));
        for narration in ["I often find myself wondering about it.", "I never expected to sing in public.", "I always thought I disliked dancing.", "I usually walked to school back then."] {
            assert!(!is_preference_sentence(narration), "{narration}");
        }
    }

    #[test]
    fn kinds_and_novelty() {
        let kinds = rule_candidates("r", "I love jazz. I usually host a potluck on Fridays. Always answer briefly.", None).into_iter().map(|c| c.kind.unwrap()).collect::<Vec<_>>();
        assert_eq!(kinds, ["likes", "does", "instruction"]);
        assert_eq!((normalize_kind("Habit").as_deref(), normalize_kind("dislikes").as_deref(), normalize_kind("mood")), (Some("does"), Some("likes"), None));
        assert!(asks_for_novelty("Any new ways to connect with friends over meals?") && asks_for_novelty("I want to try something different this weekend"));
        assert!(!asks_for_novelty("What did I say about my sister's wedding?"));
        let mut p = Preference {
            id: "p".into(), statement: "The user hosts potlucks.".into(), when_to_apply: None, category: None, evidence: None, record_id: None,
            source: "llm".into(), status: "current".into(), version: 1, replaces: None, stated_at: None, updated_at: String::new(), embedding: vec![], kind: Some("does".into()),
        };
        assert!(p.render(Labels::Novelty, true).starts_with("[the user already does this; not new to them] "));
        assert!(p.render(Labels::Novelty, false).starts_with("[the user already does this] "));
        assert!(p.render(Labels::Kind, true).starts_with("[the user already does this] "));
        assert!(p.render(Labels::Plain, true).starts_with("[preference] "));
        p.kind = Some("likes".into());
        assert!(p.render(Labels::Novelty, true).starts_with("[preference] "));
    }

    #[test]
    fn defaults_are_off_and_plain() {
        if std::env::var_os("TIDEMARK_PREFERENCES").is_none() && std::env::var_os("TIDEMARK_PREFERENCE_LABELS").is_none() {
            let c = Config::from_env().unwrap();
            assert_eq!((c.mode, c.labels, c.slots), (Mode::Off, Labels::Plain, 0));
        }
    }

    #[test]
    fn evidence_must_be_the_users_words() {
        let text = "I’m allergic to peanuts, and I usually cook at home.";
        assert!(evidence_holds("I'm allergic to peanuts", text), "quote styles and case ignored");
        assert!(evidence_holds("  usually cook at home. ", text));
        assert!(!evidence_holds("I am allergic to peanuts", text), "a paraphrase is not evidence");
        assert!(!evidence_holds("I", text), "too short to show anything");
    }

    #[test]
    fn user_texts_follow_the_records() {
        let rows = vec![
            ("b0".to_string(), "Hello\nI love tea.\nTea is nice.".to_string(), None),
            ("b1".to_string(), "assistant only".to_string(), None),
            ("p1".to_string(), "Caroline: second part of a long one".to_string(), None),
        ];
        let users = vec!["I love tea.".to_string(), "Caroline: first part ... second part of a long one".to_string()];
        let texts = user_texts(&rows, &users);
        assert_eq!(texts.iter().map(|t| (t.0.as_str(), t.1.as_str())).collect::<Vec<_>>(), [("b0", "I love tea."), ("p1", "Caroline: second part of a long one")]);
    }

    #[test]
    fn model_proposals_are_checked() {
        let batch = vec![
            Pending { record_id: "r1".into(), text: "I love trail running on weekends.".into(), stated_at: None, attempts: 0 },
            Pending { record_id: "r2".into(), text: "Actually I switched to cycling.".into(), stated_at: None, attempts: 0 },
        ];
        let current = HashSet::from(["pref_old".to_string()]);
        let reply = json!({"preferences": [
            {"record": "R1", "statement": "The user enjoys trail running.", "when_to_apply": "weekend plans", "evidence": "I love trail running", "replaces": null},
            {"record": "R2", "statement": "The user now cycles.", "evidence": "switched to cycling", "replaces": "pref_old"},
            {"record": "R2", "statement": "The user hates running.", "evidence": "I hate running"},
            {"record": "R9", "statement": "x", "evidence": "I love trail running"},
            {"record": "R1", "statement": "The user runs.", "evidence": "I love trail running", "replaces": "pref_unknown"},
            {"statement": "no record", "evidence": "cycling"}
        ]});
        let (kept, dropped) = check_reply(&reply, &batch, &current);
        assert_eq!(dropped, 3);
        let (_, verdicts) = check_reply_audited(&reply, &batch, &current);
        let reasons = verdicts.iter().map(|v| v["verdict"].as_str().unwrap()).collect::<Vec<_>>();
        assert_eq!(reasons, ["kept", "kept", "quote_not_verbatim", "unknown_record", "kept", "unknown_record"]);
        assert_eq!(kept.len(), 3);
        assert_eq!((kept[0].when_to_apply.as_deref(), kept[0].record_id.as_deref()), (Some("weekend plans"), Some("r1")));
        assert_eq!(kept[1].replaces.as_deref(), Some("pref_old"));
        assert_eq!(kept[2].replaces, None, "an unknown id is not replaced");
    }

    #[test]
    fn storing_dedupes_and_versions() {
        let mut db = Connection::open_in_memory().unwrap();
        super::super::open_schema(&db).unwrap();
        db.execute("INSERT INTO records (id, user_id, session_id, request_id, seq, raw, content, embedding) VALUES ('r1', 'u', 's', 'q', 1, 'x', 'x', x'')", []).unwrap();
        let c = |statement: &str, replaces: Option<&str>| Candidate {
            record_id: Some("r1".into()), statement: statement.into(), when_to_apply: None, category: None, evidence: None,
            stated_at: None, source: "llm", replaces: replaces.map(String::from), kind: None,
        };
        let tx = db.transaction().unwrap();
        assert_eq!(store(&tx, "u", &[(c("The user likes tea.", None), vec![1.0]), (c("the user likes  tea", None), vec![1.0])]).unwrap(), 1, "same statement once");
        tx.commit().unwrap();
        let old = load(&db, "u", false).unwrap()[0].id.clone();
        let tx = db.transaction().unwrap();
        assert_eq!(store(&tx, "u", &[(c("The user switched to coffee.", Some(&old)), vec![1.0])]).unwrap(), 1);
        tx.commit().unwrap();
        let all = load(&db, "u", true).unwrap();
        assert_eq!(all.iter().map(|p| (p.status.as_str(), p.version)).collect::<Vec<_>>(), [("superseded", 1), ("current", 2)]);
        assert_eq!(all[1].replaces.as_deref(), Some(old.as_str()));
        let tx = db.transaction().unwrap();
        assert_eq!(retract_record(&tx, "u", "r1").unwrap(), 1);
        tx.commit().unwrap();
        assert!(load(&db, "u", false).unwrap().is_empty());
    }
}
