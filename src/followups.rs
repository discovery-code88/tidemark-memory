//! Follow-ups (opt-in, `TIDEMARK_FOLLOWUPS=<n>`, per request `followups`): what else the user said, at another time,
//! about the thing a returned statement is about — a stop, a replacement, a correction, a resumption — shown with the
//! statement it relates to, each with its own date. A question about the current state repeats the wording of the
//! original statement ("Do I still go running in the mornings?" ↔ "I go running every morning"), so the original
//! wins the ranking and the later "I've stopped running altogether" is left out; this brings it along.
//!
//! Nothing is decided here: no statement is marked as overriding another, stored records are not changed, and the
//! ranking is not changed. A follow-up is attached to the returned record it relates to, so it takes no result slot.
//!
//! Rules (all from runtime evidence; no cue words, no labels):
//! - An anchor is a sentence of the user's own lines in a returned record that covers at least `anchor_share` of
//!   the question's wording (its terms weighted by how distinctive they are among the user's records). A returned
//!   sentence that only shares a common word with the question is not what the question is about.
//! - A follow-up is a sentence of the user's own lines in a record that was not returned, which shares distinctive
//!   terms (occurring in at most `rare` of the user's records) with an anchor: either the anchor's subject — the
//!   most distinctive term it shares with the question ("running", not "go" or "still") — or, when `min_terms`
//!   is set (0 = off, the default), at least that many distinctive terms of any kind ("green tea"). The second
//!   link finds more replacements and attaches more unrelated sentences; see docs/EXPERIMENTS.md.
//! - Scope `later` (default) keeps only sentences from records dated after the anchor's record; records without a
//!   date are then never follow-ups, since storage order is not time. Scope `any` also keeps earlier ones.
//! - At most `max` follow-ups per search, the most distinctive links first.

use super::*;
use std::collections::HashSet;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Options {
    pub max: usize,
    pub rare: f32,
    pub min_terms: usize,
    pub any_time: bool,
    pub anchor_share: f32,
    pub max_chars: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Followup {
    /// The record the sentence is from, and the returned record it is attached to.
    pub record: usize,
    pub anchor: usize,
    pub sentence: String,
    /// The distinctive terms shared with the anchor sentence.
    pub terms: Vec<String>,
    pub score: f32,
    /// Whether the sentence's record is dated after the anchor's.
    pub later: bool,
}

/// The user's sentences of a record, without the time notes Add appended to their lines: a note is not something
/// the user said, and on its own it is not a statement.
fn user_sentences(content: &str) -> Vec<String> {
    static NOTE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let note = NOTE.get_or_init(|| regex::Regex::new(r"\s*\[time: [^\]]*\]").unwrap());
    search::user_lines(content).into_iter().flat_map(|line| preferences::sentences(&note.replace_all(line, "")).into_iter().map(String::from).collect::<Vec<_>>()).collect()
}

/// The follow-ups for the returned records `ranked` (best first), most distinctive first.
pub fn find(records: &[Record], ranked: &[usize], query_tokens: &[String], index: &search::Bm25Index, opt: Options) -> Vec<Followup> {
    if opt.max == 0 || ranked.is_empty() || query_tokens.is_empty() {
        return Vec::new();
    }
    let n = index.len().max(1) as f32;
    let idf = |t: &str| ((n - index.df(t) as f32 + 0.5) / (index.df(t) as f32 + 0.5) + 1.0).ln();
    let is_rare = |t: &str| index.df(t) > 0 && (index.df(t) as f32) / n <= opt.rare;
    let asked = query_tokens.iter().map(String::as_str).collect::<HashSet<_>>();
    // The question's wording, among the terms the user has ever used.
    let mass = asked.iter().filter(|t| index.df(t) > 0).map(|t| idf(t)).sum::<f32>().max(f32::MIN_POSITIVE);
    // Anchors: (rank, record, distinctive terms of a user sentence that covers enough of the question, its subject).
    let mut anchors: Vec<(usize, usize, HashSet<String>, HashSet<String>)> = Vec::new();
    for (rank, &i) in ranked.iter().enumerate() {
        for sentence in user_sentences(&records[i].content) {
            let tokens = search::tokens(&sentence);
            let covered = tokens.iter().filter(|t| asked.contains(t.as_str())).map(String::as_str).collect::<HashSet<_>>();
            if covered.is_empty() || covered.iter().map(|t| idf(t)).sum::<f32>() / mass < opt.anchor_share {
                continue;
            }
            // The subject: the most distinctive asked term that another record also has (one found only here links nothing).
            let linking = |t: &&&str| is_rare(t) && index.df(t) >= 2;
            let top = covered.iter().filter(linking).map(|t| idf(t)).fold(0.0f32, f32::max);
            let subject = covered.iter().filter(linking).filter(|t| idf(t) >= top).map(|t| t.to_string()).collect::<HashSet<_>>();
            let terms = tokens.into_iter().filter(|t| is_rare(t)).collect::<HashSet<_>>();
            if !terms.is_empty() {
                anchors.push((rank, i, terms, subject));
            }
        }
    }
    if anchors.is_empty() {
        return Vec::new();
    }
    let wanted = anchors.iter().flat_map(|(_, _, t, _)| t.iter().map(String::as_str)).collect::<HashSet<_>>();
    let returned = ranked.iter().copied().collect::<HashSet<_>>();
    let mut found: Vec<(Followup, usize)> = Vec::new();
    for (j, r) in records.iter().enumerate() {
        if returned.contains(&j) || !r.user_tokens.iter().any(|t| wanted.contains(t.as_str())) {
            continue;
        }
        for sentence in user_sentences(&r.content) {
            let sentence = sentence.as_str();
            let tokens = search::tokens(sentence).into_iter().collect::<HashSet<_>>();
            let mut best: Option<(Followup, usize)> = None;
            for (rank, a, terms, subject) in &anchors {
                let later = match (&r.created_at, &records[*a].created_at) {
                    (Some(c), Some(d)) => c > d,
                    _ => false,
                };
                if !later && (!opt.any_time || r.created_at == records[*a].created_at && r.created_at.is_some()) {
                    continue;
                }
                let mut shared = tokens.iter().filter(|t| terms.contains(*t)).cloned().collect::<Vec<_>>();
                if shared.is_empty() || !(opt.min_terms > 0 && shared.len() >= opt.min_terms || shared.iter().any(|t| subject.contains(t))) {
                    continue;
                }
                shared.sort();
                // The subject counts double: the link is then about what was asked.
                let score = shared.iter().map(|t| idf(t) * if subject.contains(t) { 2.0 } else { 1.0 }).sum::<f32>();
                if best.as_ref().is_none_or(|(b, br)| score > b.score || score == b.score && rank < br) {
                    let text = sentence.chars().take(opt.max_chars.max(1)).collect::<String>();
                    best = Some((Followup { record: j, anchor: *a, sentence: text, terms: shared, score, later }, *rank));
                }
            }
            if let Some(b) = best {
                // A sentence the caller already receives is not a follow-up.
                if !ranked.iter().any(|&i| records[i].content.contains(sentence)) && !found.iter().any(|(f, _)| f.sentence == b.0.sentence) {
                    found.push(b);
                }
            }
        }
    }
    found.sort_by(|(a, ar), (b, br)| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then(ar.cmp(br)).then(a.record.cmp(&b.record)));
    found.into_iter().take(opt.max).map(|(f, _)| f).collect()
}

/// The line attached to the anchor's content: the follow-up with its record's date header and its relation in time,
/// and the time notes of that sentence alone, resolved against its own record's day (the sender's, when the record
/// states an offset). Notes of other sentences of the same message do not travel with it.
pub fn line(records: &[Record], f: &Followup) -> String {
    let r = &records[f.record];
    let date = r.content.strip_prefix('[').and_then(|c| c.split_once(']')).map(|(h, _)| format!(", {h}")).unwrap_or_default();
    let notes = applies::said_in(&r.content, r.created_at.as_deref()).map(|day| dates::annotate(&f.sentence, day)).unwrap_or_default();
    let notes = if notes.is_empty() { String::new() } else { format!(" [time: {}]", notes.join("; ")) };
    format!("[said {}{date}] user: {}{notes}", if f.later { "later" } else { "at another time" }, f.sentence)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: &str, at: Option<&str>, content: &str) -> Record {
        let (user_tokens, instructions, changes) = user_parts(content);
        Record { id: id.into(), speaker: None, content: content.into(), created_at: at.map(String::from), tokens: search::tokens(content), embedding: vec![], user_tokens, instructions, changes }
    }

    fn index(records: &[Record]) -> search::Bm25Index {
        let mut index = search::Bm25Index::default();
        for r in records {
            index.add(&r.user_tokens);
        }
        index
    }

    fn history() -> Vec<Record> {
        let mut records = vec![
            rec("run", Some("2023-03-02T12:00:00Z"), "[Thursday 2 March 2023, 12:00] user: I go running every morning before work, it's how I start my day.\nassistant: Running is a great habit."),
            rec("stop", Some("2023-06-14T12:00:00Z"), "[Wednesday 14 June 2023, 12:00] user: The weather was fine. I've stopped running altogether because my knee can't take it anymore.\nassistant: You stopped running; rest that knee."),
            rec("tea", Some("2023-02-14T12:00:00Z"), "[Tuesday 14 February 2023, 12:00] user: I drink a pot of green tea every morning."),
            rec("coffee", Some("2023-06-30T12:00:00Z"), "[Friday 30 June 2023, 12:00] user: I've switched from green tea to black coffee in the mornings."),
            rec("undated", None, "user: Running shoes are expensive."),
        ];
        for k in 0..20 {
            records.push(rec(&format!("h{k}"), Some("2023-05-01T12:00:00Z"), &format!("[Monday 1 May 2023, 12:00] user: Topic {k} is about gardens, books and travel plans number {k}.")));
        }
        records
    }

    const ON: Options = Options { max: 3, rare: 0.15, min_terms: 0, any_time: false, anchor_share: 0.5, max_chars: 400 };

    #[test]
    fn a_later_stop_follows_the_statement_the_question_echoes() {
        let records = history();
        let idx = index(&records);
        let found = find(&records, &[0, 5], &search::tokens("Do I still go running in the mornings?"), &idx, ON);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!((records[found[0].record].id.as_str(), records[found[0].anchor].id.as_str(), found[0].later), ("stop", "run", true));
        assert_eq!(found[0].sentence, "I've stopped running altogether because my knee can't take it anymore.", "the user's sentence only, not the reply or its neighbours");
        assert_eq!(line(&records, &found[0]), "[said later, Wednesday 14 June 2023, 12:00] user: I've stopped running altogether because my knee can't take it anymore.");
    }

    #[test]
    fn a_follow_up_carries_its_own_sentences_notes_and_no_other() {
        let mut records = history();
        records[1] = rec("stop", Some("2023-06-14T22:30:00Z"), "[Thursday 15 June 2023, 00:30 UTC+02:00] user: Ana told me she moved last month. I stopped running yesterday because of my knee. [time: last month = May 2023; yesterday = 14 June 2023]\nassistant: Rest that knee.");
        let idx = index(&records);
        let found = find(&records, &[0, 5], &search::tokens("Do I still go running in the mornings?"), &idx, ON);
        assert_eq!(found.len(), 1, "{found:?}");
        assert_eq!(found[0].sentence, "I stopped running yesterday because of my knee.", "the sentence without the message's notes");
        assert_eq!(
            line(&records, &found[0]),
            "[said later, Thursday 15 June 2023, 00:30 UTC+02:00] user: I stopped running yesterday because of my knee. [time: yesterday = 14 June 2023]",
            "its own note, on the sender's day; the note about Ana's move stays with Ana's sentence"
        );
        assert!(found.iter().all(|f| !f.sentence.contains("[time:")), "a note alone is never a follow-up");
    }

    #[test]
    fn two_distinctive_terms_link_without_the_question_naming_them() {
        let records = history();
        let idx = index(&records);
        let q = search::tokens("What do I drink in the mornings?");
        assert!(find(&records, &[2], &q, &idx, ON).iter().all(|f| records[f.record].id != "coffee" || f.terms != ["green", "tea"]), "off by default");
        let found = find(&records, &[2], &q, &idx, Options { min_terms: 2, ..ON });
        assert!(found.iter().any(|f| records[f.record].id == "coffee" && f.terms == ["green", "tea"]), "{found:?}");
    }

    #[test]
    fn earlier_and_undated_statements_need_the_wider_scope() {
        let records = history();
        let idx = index(&records);
        let q = search::tokens("Am I running?");
        assert!(find(&records, &[1], &q, &idx, ON).is_empty(), "the earlier statement and the undated one are not later");
        let any = find(&records, &[1], &q, &idx, Options { any_time: true, ..ON });
        assert_eq!(any.iter().map(|f| (records[f.record].id.as_str(), f.later)).collect::<Vec<_>>(), [("run", false), ("undated", false)]);
    }

    #[test]
    fn nothing_is_added_when_off_returned_already_or_unrelated() {
        let records = history();
        let idx = index(&records);
        let q = search::tokens("Do I still go running in the mornings?");
        assert!(find(&records, &[0], &q, &idx, Options { max: 0, ..ON }).is_empty(), "off");
        assert!(find(&records, &[0, 1], &q, &idx, ON).is_empty(), "the stop is already among the results");
        assert!(find(&records, &[6, 7], &search::tokens("Which topic is about gardens?"), &idx, ON).is_empty(), "terms in most records are not distinctive");
        assert!(find(&records, &[0], &search::tokens("What is the capital of France?"), &idx, ON).is_empty(), "no returned sentence shares a term with the question");
        assert!(find(&records, &[0], &search::tokens("Which museum exhibitions open on a weekday morning?"), &idx, ON).is_empty(), "a sentence sharing one minor word with the question is not an anchor");
    }
}
